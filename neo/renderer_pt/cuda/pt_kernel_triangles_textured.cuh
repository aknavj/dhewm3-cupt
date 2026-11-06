#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"
#include "renderer_pt/cuda/pt_cuda_material.cuh"

/*
========================
IntersectTriangleUV_Simple
Möller-Trumbore with barycentric output, returns t (or -1 on miss)
========================
*/
__device__ __forceinline__ float IntersectTriangleUV_Simple(
	float origX, float origY, float origZ,
	float dirX, float dirY, float dirZ,
	const ptVertex_t* vertices,
	const ptTriangle_t& tri,
	float& outU, float& outV
) {
	const float* v0 = vertices[tri.vertexIndices[0]].position;
	const float* v1 = vertices[tri.vertexIndices[1]].position;
	const float* v2 = vertices[tri.vertexIndices[2]].position;

	float e1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
	float e2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };

	float d[3] = { dirX, dirY, dirZ };
	float h[3];
	cross3(d, e2, h);
	float a = dot3(e1, h);

	if (fabsf(a) < 1e-7f) return -1.0f;

	float f = 1.0f / a;
	float s[3] = { origX - v0[0], origY - v0[1], origZ - v0[2] };
	float u = f * dot3(s, h);
	if (u < 0.0f || u > 1.0f) return -1.0f;

	float q[3];
	cross3(s, e1, q);
	float v = f * dot3(d, q);
	if (v < 0.0f || u + v > 1.0f) return -1.0f;

	float t = f * dot3(e2, q);
	if (t > 1e-4f) {
		outU = u;
		outV = v;
		return t;
	}
	return -1.0f;
}

/*
========================
TriangleDrawTexturedKernel
========================
*/
__global__ void TriangleDrawTexturedKernel(
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* bvhNodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	const ptLight_t* lights,
	int numLights,
	float* framebuffer,
	int texSubMode,
	const ptRenderParams_t params
) {
	PT_UNPACK_RENDER_PARAMS(params);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = (y * width + x) * 4;

	const int subMode = texSubMode;

	// generate primary ray
	float halfTanX = tanf(fov * 0.5f * 3.14159265f / 180.0f);
	float halfTanY = tanf(fovY * 0.5f * 3.14159265f / 180.0f);

	float u = (2.0f * ((float)x + 0.5f) / (float)width  - 1.0f) * halfTanX;
	float v = (2.0f * ((float)y + 0.5f) / (float)height - 1.0f) * halfTanY;

	float dirX = cameraForward[0] + u * cameraRight[0] + v * cameraUp[0];
	float dirY = cameraForward[1] + u * cameraRight[1] + v * cameraUp[1];
	float dirZ = cameraForward[2] + u * cameraRight[2] + v * cameraUp[2];

	float invLen = rsqrtf(dirX * dirX + dirY * dirY + dirZ * dirZ);
	dirX *= invLen;
	dirY *= invLen;
	dirZ *= invLen;

	float origX = cameraPos[0];
	float origY = cameraPos[1];
	float origZ = cameraPos[2];

	float invDirX = 1.0f / (fabsf(dirX) > 1e-8f ? dirX : copysignf(1e-8f, dirX));
	float invDirY = 1.0f / (fabsf(dirY) > 1e-8f ? dirY : copysignf(1e-8f, dirY));
	float invDirZ = 1.0f / (fabsf(dirZ) > 1e-8f ? dirZ : copysignf(1e-8f, dirZ));

	// BVH traversal — find nearest triangle with barycentrics
	float nearestT = 1e30f;
	int hitTriIdx = -1;
	float hitU = 0.0f;
	float hitV = 0.0f;

	for (int nodeIdx = 0; nodeIdx >= 0; ) {
		const ptBVHNode_t& node = bvhNodes[nodeIdx];

		if (!IntersectAABB_Simple(origX, origY, origZ, invDirX, invDirY, invDirZ,
								  node.bounds, nearestT)) {
			nodeIdx = node.escapeIndex;
			continue;
		}

		if (node.leftChild == -1) {
			// Leaf node
			for (int i = 0; i < node.primitiveCount; i++) {
				int triIdx = triIndices[node.firstPrimitive + i];
				float bu, bv;
				float t = IntersectTriangleUV_Simple(origX, origY, origZ,
													  dirX, dirY, dirZ,
													  vertices, triangles[triIdx],
													  bu, bv);
				if (t > 0.0f && t < nearestT) {
					nearestT = t;
					hitTriIdx = triIdx;
					hitU = bu;
					hitV = bv;
				}
			}
			nodeIdx = node.escapeIndex;
		} else {
			nodeIdx = node.leftChild;
		}
	}

	// shade hit point
	float r, g, b;

	if (hitTriIdx >= 0) {
		const ptTriangle_t& tri = triangles[hitTriIdx];
		int i0 = tri.vertexIndices[0];
		int i1 = tri.vertexIndices[1];
		int i2 = tri.vertexIndices[2];

		float w0 = 1.0f - hitU - hitV;
		float w1 = hitU;
		float w2 = hitV;

		// interpolate texcoords
		float tu = w0 * vertices[i0].texcoord[0] + w1 * vertices[i1].texcoord[0] + w2 * vertices[i2].texcoord[0];
		float tv = w0 * vertices[i0].texcoord[1] + w1 * vertices[i1].texcoord[1] + w2 * vertices[i2].texcoord[1];

		const ptMaterial_t& mat = materials[tri.materialIndex];

		if (subMode == 1) {
			// base color only
			r = mat.albedo[0];
			g = mat.albedo[1];
			b = mat.albedo[2];

		} else if (subMode == 2) {
			// albedo texture only
			if (mat.albedoTexture >= 0) {
				float color[4];
				SampleTexture(textures, mat.albedoTexture, tu, tv, color);
				r = color[0];
				g = color[1];
				b = color[2];
			} else {
				r = 1.0f; g = 0.0f; b = 1.0f; // magenta = no texture
			}

		} else if (subMode == 3) {
			// normal texture
			if (mat.normalTexture >= 0) {
				float color[4];
				SampleTexture(textures, mat.normalTexture, tu, tv, color);
				// normal maps are stored as tangent-space XYZ in RGB
				r = color[0];
				g = color[1];
				b = color[2];
			} else {
				// no normal map — show interpolated geometric normal as color
				float nx = w0 * vertices[i0].normal[0] + w1 * vertices[i1].normal[0] + w2 * vertices[i2].normal[0];
				float ny = w0 * vertices[i0].normal[1] + w1 * vertices[i1].normal[1] + w2 * vertices[i2].normal[1];
				float nz = w0 * vertices[i0].normal[2] + w1 * vertices[i1].normal[2] + w2 * vertices[i2].normal[2];
				float nLen = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
				r = nx * nLen * 0.5f + 0.5f;
				g = ny * nLen * 0.5f + 0.5f;
				b = nz * nLen * 0.5f + 0.5f;
			}

		} else if (subMode == 4) {
			// specular texture
			if (mat.specularTexture >= 0) {
				float color[4];
				SampleTexture(textures, mat.specularTexture, tu, tv, color);
				r = color[0];
				g = color[1];
				b = color[2];
			} else {
				// no specular map — show roughness as grayscale
				float rough = mat.roughness;
				r = rough;
				g = rough;
				b = rough;
			}

		} else {
			// subMode 0 (default): Combined (albedo * base color, Lambertian lit)
			float albedoR = mat.albedo[0];
			float albedoG = mat.albedo[1];
			float albedoB = mat.albedo[2];

			if (mat.albedoTexture >= 0) {
				float color[4];
				SampleTexture(textures, mat.albedoTexture, tu, tv, color);
				albedoR *= color[0];
				albedoG *= color[1];
				albedoB *= color[2];
			}

			// interpolate normal for Lambertian shading
			float nx = w0 * vertices[i0].normal[0] + w1 * vertices[i1].normal[0] + w2 * vertices[i2].normal[0];
			float ny = w0 * vertices[i0].normal[1] + w1 * vertices[i1].normal[1] + w2 * vertices[i2].normal[1];
			float nz = w0 * vertices[i0].normal[2] + w1 * vertices[i1].normal[2] + w2 * vertices[i2].normal[2];
			float nInvLen = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
			nx *= nInvLen;
			ny *= nInvLen;
			nz *= nInvLen;

			// fixed directional light for simple shading
			float lightDirX = 0.577f, lightDirY = 0.577f, lightDirZ = 0.577f;
			float NdotL = fmaxf(0.0f, nx * lightDirX + ny * lightDirY + nz * lightDirZ);

			float lighting = NdotL;

			albedoR *= lighting;
			albedoG *= lighting;
			albedoB *= lighting;

			r = fminf(albedoR, 1.0f);
			g = fminf(albedoG, 1.0f);
			b = fminf(albedoB, 1.0f);
		}
	} else {
		// sky background (scaled by skyIntensity, defaults to black)
		r = skyColorHorizon[0] * skyIntensity;
		g = skyColorHorizon[1] * skyIntensity;
		b = skyColorHorizon[2] * skyIntensity;
	}

	// write HDR framebuffer (float4 RGBA)
	framebuffer[pixelIndex + 0] = r;
	framebuffer[pixelIndex + 1] = g;
	framebuffer[pixelIndex + 2] = b;
	framebuffer[pixelIndex + 3] = 1.0f;
}
