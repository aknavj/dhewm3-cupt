#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"

/*
========================
IntersectAABB_Simple
Ray-AABB test for BVH traversal
========================
*/
__device__ __forceinline__ bool IntersectAABB_Simple(
	float origX, float origY, float origZ,
	float invDirX, float invDirY, float invDirZ,
	const float* bounds,
	float maxT
) {
	float t1 = (bounds[0] - origX) * invDirX;
	float t2 = (bounds[3] - origX) * invDirX;
	float tmin = fminf(t1, t2);
	float tmax = fmaxf(t1, t2);

	t1 = (bounds[1] - origY) * invDirY;
	t2 = (bounds[4] - origY) * invDirY;
	tmin = fmaxf(tmin, fminf(t1, t2));
	tmax = fminf(tmax, fmaxf(t1, t2));

	t1 = (bounds[2] - origZ) * invDirZ;
	t2 = (bounds[5] - origZ) * invDirZ;
	tmin = fmaxf(tmin, fminf(t1, t2));
	tmax = fminf(tmax, fmaxf(t1, t2));

	return tmax >= fmaxf(tmin, 0.0f) && tmin < maxT;
}

/*
========================
IntersectTriangle_Simple
Möller-Trumbore intersection, returns t (or -1 on miss)
========================
*/
__device__ __forceinline__ float IntersectTriangle_Simple(
	float origX, float origY, float origZ,
	float dirX, float dirY, float dirZ,
	const ptVertex_t* vertices,
	const ptTriangle_t& tri
) {
	const float* v0 = vertices[tri.vertexIndices[0]].position;
	const float* v1 = vertices[tri.vertexIndices[1]].position;
	const float* v2 = vertices[tri.vertexIndices[2]].position;

	float e1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
	float e2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };

	float h[3];
	float d[3] = { dirX, dirY, dirZ };
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
	return (t > 1e-4f) ? t : -1.0f;
}

/*
========================
TriangleDrawKernel
Simple BVH debug visualization — random color per triangle.
========================
*/
__global__ void TriangleDrawKernel(
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* bvhNodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	const ptLight_t* lights,
	int numLights,
	float* framebuffer,
	const ptRenderParams_t params
) {
	PT_UNPACK_RENDER_PARAMS(params);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = (y * width + x) * 4;

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

	// BVH traversal — find nearest triangle
	float nearestT = 1e30f;
	int hitTriIdx = -1;

	for (int nodeIdx = 0; nodeIdx >= 0; ) {
		const ptBVHNode_t& node = bvhNodes[nodeIdx];

		if (!IntersectAABB_Simple(origX, origY, origZ, invDirX, invDirY, invDirZ,
								  node.bounds, nearestT)) {
			nodeIdx = node.escapeIndex;
			continue;
		}

		if (node.leftChild == -1) {
			// leaf node
			for (int i = 0; i < node.primitiveCount; i++) {
				int triIdx = triIndices[node.firstPrimitive + i];
				float t = IntersectTriangle_Simple(origX, origY, origZ,
												   dirX, dirY, dirZ,
												   vertices, triangles[triIdx]);
				if (t > 0.0f && t < nearestT) {
					nearestT = t;
					hitTriIdx = triIdx;
				}
			}
			nodeIdx = node.escapeIndex;
		} else {
			nodeIdx = node.leftChild;
		}
	}

	// color output — random color per triangle
	float r, g, b;

	if (hitTriIdx >= 0) {
		unsigned int hash = cupt_hash((unsigned int)hitTriIdx * 7919u + 1u);
		r = float((hash >>  0) & 0xFF) / 255.0f;
		g = float((hash >>  8) & 0xFF) / 255.0f;
		b = float((hash >> 16) & 0xFF) / 255.0f;

		r = r * 0.7f + 0.3f;
		g = g * 0.7f + 0.3f;
		b = b * 0.7f + 0.3f;
	} else {
		r = 0.0f;
		g = 0.0f;
		b = 0.0f;
	}

	// write HDR framebuffer (float4 RGBA)
	framebuffer[pixelIndex + 0] = r;
	framebuffer[pixelIndex + 1] = g;
	framebuffer[pixelIndex + 2] = b;
	framebuffer[pixelIndex + 3] = 1.0f;
}