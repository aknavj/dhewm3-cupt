#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"
#include "renderer_pt/cuda/pt_cuda_material.cuh"

/*
=================================================================================
Ray-Triangle Intersection (Möller-Trumbore algorithm)
=================================================================================
*/
__device__ bool IntersectTriangle(
	const ptRay_t& ray,
	const ptVertex_t* vertices,
	const ptTriangle_t& triangle,
	const ptMaterial_t* materials,
	ptHitInfo_t& hitInfo
) {
	const ptVertex_t& v0 = vertices[triangle.vertexIndices[0]];
	const ptVertex_t& v1 = vertices[triangle.vertexIndices[1]];
	const ptVertex_t& v2 = vertices[triangle.vertexIndices[2]];

	float edge1[3] = {
		v1.position[0] - v0.position[0],
		v1.position[1] - v0.position[1],
		v1.position[2] - v0.position[2]
	};
	float edge2[3] = {
		v2.position[0] - v0.position[0],
		v2.position[1] - v0.position[1],
		v2.position[2] - v0.position[2]
	};

	float h[3];
	cross3(ray.direction, edge2, h);
	float a = dot3(edge1, h);

	if (fabsf(a) < 1e-6f) return false;

	float f = 1.0f / a;
	float s[3] = {
		ray.origin[0] - v0.position[0],
		ray.origin[1] - v0.position[1],
		ray.origin[2] - v0.position[2]
	};

	float u = f * dot3(s, h);
	if (u < 0.0f || u > 1.0f) return false;

	float q[3];
	cross3(s, edge1, q);
	float v = f * dot3(ray.direction, q);
	if (v < 0.0f || u + v > 1.0f) return false;

	float t = f * dot3(edge2, q);
	if (t > ray.tMin && t < ray.tMax && t < hitInfo.t) {
		hitInfo.hit = true;
		hitInfo.t = t;

		hitInfo.position[0] = ray.origin[0] + t * ray.direction[0];
		hitInfo.position[1] = ray.origin[1] + t * ray.direction[1];
		hitInfo.position[2] = ray.origin[2] + t * ray.direction[2];

		float w = 1.0f - u - v;
		hitInfo.normal[0] = w * v0.normal[0] + u * v1.normal[0] + v * v2.normal[0];
		hitInfo.normal[1] = w * v0.normal[1] + u * v1.normal[1] + v * v2.normal[1];
		hitInfo.normal[2] = w * v0.normal[2] + u * v1.normal[2] + v * v2.normal[2];
		normalize3(hitInfo.normal);

		hitInfo.tangent[0] = w * v0.tangent[0] + u * v1.tangent[0] + v * v2.tangent[0];
		hitInfo.tangent[1] = w * v0.tangent[1] + u * v1.tangent[1] + v * v2.tangent[1];
		hitInfo.tangent[2] = w * v0.tangent[2] + u * v1.tangent[2] + v * v2.tangent[2];
		normalize3(hitInfo.tangent);

		hitInfo.binormal[0] = w * v0.binormal[0] + u * v1.binormal[0] + v * v2.binormal[0];
		hitInfo.binormal[1] = w * v0.binormal[1] + u * v1.binormal[1] + v * v2.binormal[1];
		hitInfo.binormal[2] = w * v0.binormal[2] + u * v1.binormal[2] + v * v2.binormal[2];
		normalize3(hitInfo.binormal);

		hitInfo.texcoord[0] = w * v0.texcoord[0] + u * v1.texcoord[0] + v * v2.texcoord[0];
		hitInfo.texcoord[1] = w * v0.texcoord[1] + u * v1.texcoord[1] + v * v2.texcoord[1];

		hitInfo.barycentrics[0] = w;
		hitInfo.barycentrics[1] = u;
		hitInfo.barycentrics[2] = v;

		hitInfo.materialIndex = triangle.materialIndex;

		return true;
	}

	return false;
}

/*
=================================================================================
BVH Traversal
=================================================================================
*/
__device__ bool IntersectBVH(
	const ptRay_t& ray,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* nodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	ptHitInfo_t& hitInfo
) {
	bool foundHit = false;

	int stack[128];
	int stackPtr = 0;
	stack[stackPtr++] = 0;

	while (stackPtr > 0 && stackPtr < 128) {
		int nodeIndex = stack[--stackPtr];
		const ptBVHNode_t& node = nodes[nodeIndex];

		float tmin = ray.tMin;
		float tmax = foundHit ? hitInfo.t : ray.tMax;

		for (int i = 0; i < 3; i++) {
			if (fabsf(ray.direction[i]) < 1e-6f) {
				if (ray.origin[i] < node.bounds[i] || ray.origin[i] > node.bounds[i + 3]) {
					goto skip_node;
				}
				continue;
			}

			float invD = 1.0f / ray.direction[i];
			float t0 = (node.bounds[i] - ray.origin[i]) * invD;
			float t1 = (node.bounds[i + 3] - ray.origin[i]) * invD;

			if (invD < 0.0f) {
				float tmp = t0; t0 = t1; t1 = tmp;
			}

			tmin = fmaxf(tmin, t0);
			tmax = fminf(tmax, t1);

			if (tmax <= tmin) goto skip_node;
		}

		if (node.leftChild == -1) {
			for (int i = node.firstPrimitive; i < node.firstPrimitive + node.primitiveCount; i++) {
				int triIdx = triIndices[i];
				if (IntersectTriangle(ray, vertices, triangles[triIdx], materials, hitInfo)) {
					hitInfo.triangleIndex = triIdx;
					foundHit = true;
				}
			}
		} else {
			if (stackPtr <= 126) {
				const ptBVHNode_t& leftNode = nodes[node.leftChild];
				const ptBVHNode_t& rightNode = nodes[node.rightChild];

				float leftDot =
					((leftNode.bounds[0] + leftNode.bounds[3]) - 2.0f * ray.origin[0]) * ray.direction[0] +
					((leftNode.bounds[1] + leftNode.bounds[4]) - 2.0f * ray.origin[1]) * ray.direction[1] +
					((leftNode.bounds[2] + leftNode.bounds[5]) - 2.0f * ray.origin[2]) * ray.direction[2];
				float rightDot =
					((rightNode.bounds[0] + rightNode.bounds[3]) - 2.0f * ray.origin[0]) * ray.direction[0] +
					((rightNode.bounds[1] + rightNode.bounds[4]) - 2.0f * ray.origin[1]) * ray.direction[1] +
					((rightNode.bounds[2] + rightNode.bounds[5]) - 2.0f * ray.origin[2]) * ray.direction[2];

				if (leftDot < rightDot) {
					stack[stackPtr++] = node.rightChild;
					stack[stackPtr++] = node.leftChild;
				} else {
					stack[stackPtr++] = node.leftChild;
					stack[stackPtr++] = node.rightChild;
				}
			}
		}

		skip_node:;
	}

	return foundHit;
}

// include sampling after intersection is defined (sampling calls IntersectBVH)
#include "renderer_pt/cuda/pt_cuda_sampling.cuh"

/*
=================================================================================
PathTracingKernel - Full PBR path tracing
=================================================================================
*/
__global__ void PathTracingKernel(
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

	if (x >= width || y >= height) return;

	int pixelIndex = (y * width + x) * 4;

	// per-pixel RNG with curand
	unsigned long long pixelSeed = (unsigned long long)(y * width + x) * 1000003ULL + (unsigned long long)frameIndex * 999983ULL;
	curandState randState;
	curand_init(pixelSeed, 0, 0, &randState);

	float color[3] = { 0.0f, 0.0f, 0.0f };

	for (int sample = 0; sample < samplesPerPixel; sample++) {
		// primary ray with jitter
		float jitterX = curand_uniform(&randState) - 0.5f;
		float jitterY = curand_uniform(&randState) - 0.5f;

		float u = ((float)(x) + 0.5f + jitterX) / (float)width * 2.0f - 1.0f;
		float v = ((float)(y) + 0.5f + jitterY) / (float)height * 2.0f - 1.0f;

		float tanHalfFovX = tanf(fov * PT_PI / 360.0f);
		float tanHalfFovY = tanf(fovY * PT_PI / 360.0f);

		ptRay_t ray;
		ray.origin[0] = cameraPos[0];
		ray.origin[1] = cameraPos[1];
		ray.origin[2] = cameraPos[2];

		ray.direction[0] = cameraForward[0] + u * cameraRight[0] * tanHalfFovX + v * cameraUp[0] * tanHalfFovY;
		ray.direction[1] = cameraForward[1] + u * cameraRight[1] * tanHalfFovX + v * cameraUp[1] * tanHalfFovY;
		ray.direction[2] = cameraForward[2] + u * cameraRight[2] * tanHalfFovX + v * cameraUp[2] * tanHalfFovY;
		normalize3(ray.direction);

		ray.tMin = rayOffset;
		ray.tMax = 1e10f;

		float pathThroughput[3] = { 1.0f, 1.0f, 1.0f };
		float pathRadiance[3] = { 0.0f, 0.0f, 0.0f };

		int depth = 0;
		const int maxTransparentPasses = maxDepth * 8;  // generous budget for overlapping particles/decals
		for (int iteration = 0; iteration < maxTransparentPasses && depth < maxDepth; iteration++) {
			ptHitInfo_t hitInfo;
			hitInfo.hit = false;
			hitInfo.t = ray.tMax;
			hitInfo.materialIndex = 0;
			hitInfo.triangleIndex = 0;
			hitInfo.barycentrics[0] = 0.0f;
			hitInfo.barycentrics[1] = 0.0f;
			hitInfo.barycentrics[2] = 0.0f;

			if (!IntersectBVH(ray, vertices, triangles, bvhNodes, triIndices, materials, hitInfo)) {
				// sky dome - primary rays only (bounce misses return black for Doom 3 darkness)
				if (depth == 0 && skyIntensity > 0.0f) {
					float upDot = ray.direction[2];
					float skyColor[3];
					if (upDot > 0.0f) {
						float t = fminf(1.0f, upDot * 2.0f);
						skyColor[0] = skyColorHorizon[0] * (1.0f - t) + skyColorZenith[0] * t;
						skyColor[1] = skyColorHorizon[1] * (1.0f - t) + skyColorZenith[1] * t;
						skyColor[2] = skyColorHorizon[2] * (1.0f - t) + skyColorZenith[2] * t;
					} else {
						float t = fminf(1.0f, -upDot * 3.0f);
						skyColor[0] = skyColorHorizon[0] * (1.0f - t) + skyColorGround[0] * t;
						skyColor[1] = skyColorHorizon[1] * (1.0f - t) + skyColorGround[1] * t;
						skyColor[2] = skyColorHorizon[2] * (1.0f - t) + skyColorGround[2] * t;
					}
					pathRadiance[0] += pathThroughput[0] * skyColor[0] * skyIntensity;
					pathRadiance[1] += pathThroughput[1] * skyColor[1] * skyIntensity;
					pathRadiance[2] += pathThroughput[2] * skyColor[2] * skyIntensity;
				}
				break;
			}

			const ptMaterial_t& material = materials[hitInfo.materialIndex];

			// ---------------------------------------------------------------
			// Apply texture transforms to UV
			// ---------------------------------------------------------------
			float tu = hitInfo.texcoord[0];
			float tv = hitInfo.texcoord[1];

			bool hasTransform = (material.texTransform[0] != 1.0f || material.texTransform[1] != 1.0f ||
								material.texTransform[2] != 0.0f || material.texTransform[3] != 0.0f ||
								material.texTransform[4] != 0.0f);

			if (hasTransform) {
				tu *= material.texTransform[0];
				tv *= material.texTransform[1];
				if (material.texTransform[2] != 0.0f) {
					float angle = material.texTransform[2];
					float cosA = cosf(angle);
					float sinA = sinf(angle);
					float uRot = tu * cosA - tv * sinA;
					float vRot = tu * sinA + tv * cosA;
					tu = uRot;
					tv = vRot;
				}
				tu += material.texTransform[3];
				tv += material.texTransform[4];
				tu = tu - floorf(tu);
				tv = tv - floorf(tv);
			}

			// ---------------------------------------------------------------
			// Sample albedo texture
			// ---------------------------------------------------------------
			float albedo[4];
			if (material.albedoTexture >= 0 && textures) {
				SampleTexture(textures, material.albedoTexture, tu, tv, albedo);
				albedo[0] *= material.albedo[0];
				albedo[1] *= material.albedo[1];
				albedo[2] *= material.albedo[2];
				albedo[3] *= material.albedo[3];
			} else {
				albedo[0] = material.albedo[0];
				albedo[1] = material.albedo[1];
				albedo[2] = material.albedo[2];
				albedo[3] = material.albedo[3];
			}

			// ---------------------------------------------------------------
			// Normal mapping with RXGB format & degenerate tangent frame
			// ---------------------------------------------------------------
			float surfaceNormal[3];
			surfaceNormal[0] = hitInfo.normal[0];
			surfaceNormal[1] = hitInfo.normal[1];
			surfaceNormal[2] = hitInfo.normal[2];

			if (material.normalTexture >= 0 && textures) {
				float normalMap[4];
				SampleTexture(textures, material.normalTexture, tu, tv, normalMap);

				// RXGB format: A=X, G=Y, B=Z
				float tangentNormal[3];
				tangentNormal[0] = (normalMap[3] * 2.0f - 1.0f) * material.bumpScale;
				tangentNormal[1] = (normalMap[1] * 2.0f - 1.0f) * material.bumpScale;
				tangentNormal[2] = (normalMap[2] * 2.0f - 1.0f);

				float T[3] = { hitInfo.tangent[0], hitInfo.tangent[1], hitInfo.tangent[2] };
				float B[3] = { hitInfo.binormal[0], hitInfo.binormal[1], hitInfo.binormal[2] };
				float N[3] = { hitInfo.normal[0], hitInfo.normal[1], hitInfo.normal[2] };

				// Fabricate tangent frame if degenerate
				float tLen = T[0] * T[0] + T[1] * T[1] + T[2] * T[2];
				float bLen = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];

				if (tLen < 0.001f || bLen < 0.001f) {
					if (fabsf(N[2]) < 0.999f) {
						float invLen = 1.0f / sqrtf(N[0] * N[0] + N[1] * N[1]);
						T[0] = -N[1] * invLen; T[1] = N[0] * invLen; T[2] = 0.0f;
					} else {
						T[0] = 1.0f; T[1] = 0.0f; T[2] = 0.0f;
					}
					cross3(N, T, B);
				}

				surfaceNormal[0] = T[0] * tangentNormal[0] + B[0] * tangentNormal[1] + N[0] * tangentNormal[2];
				surfaceNormal[1] = T[1] * tangentNormal[0] + B[1] * tangentNormal[1] + N[1] * tangentNormal[2];
				surfaceNormal[2] = T[2] * tangentNormal[0] + B[2] * tangentNormal[1] + N[2] * tangentNormal[2];

				float len = sqrtf(surfaceNormal[0] * surfaceNormal[0] + surfaceNormal[1] * surfaceNormal[1] + surfaceNormal[2] * surfaceNormal[2]);
				if (len > 0.001f) {
					surfaceNormal[0] /= len;
					surfaceNormal[1] /= len;
					surfaceNormal[2] /= len;
				}
			}

			// ---------------------------------------------------------------
			// Specular map -> PBR roughness / metallic / spec color
			// ---------------------------------------------------------------
			float metallic = material.metallic;
			float roughness = material.roughness;
			float specColor[3] = { 1.0f, 1.0f, 1.0f };

			if (material.specularTexture >= 0 && textures) {
				float specularMap[4];
				SampleTexture(textures, material.specularTexture, tu, tv, specularMap);

				specColor[0] = specularMap[0];
				specColor[1] = specularMap[1];
				specColor[2] = specularMap[2];

				float specLum = specularMap[0] * 0.2126f + specularMap[1] * 0.7152f + specularMap[2] * 0.0722f;
				roughness *= (1.0f - specLum * 0.6f);
				metallic = fmaxf(metallic, specLum * 0.3f);
				metallic = fminf(1.0f, fmaxf(0.0f, metallic));
				roughness = fminf(1.0f, fmaxf(0.01f, roughness));
			}

			// ---------------------------------------------------------------
			// Vertex color
			// ---------------------------------------------------------------
			if (material.useVertexColor) {
				const ptTriangle_t& tri = triangles[hitInfo.triangleIndex];
				const ptVertex_t& vert0 = vertices[tri.vertexIndices[0]];
				const ptVertex_t& vert1 = vertices[tri.vertexIndices[1]];
				const ptVertex_t& vert2 = vertices[tri.vertexIndices[2]];

				float bw = hitInfo.barycentrics[0];
				float bu = hitInfo.barycentrics[1];
				float bv = hitInfo.barycentrics[2];

				float vertexColor[4];
				vertexColor[0] = bw * vert0.color[0] + bu * vert1.color[0] + bv * vert2.color[0];
				vertexColor[1] = bw * vert0.color[1] + bu * vert1.color[1] + bv * vert2.color[1];
				vertexColor[2] = bw * vert0.color[2] + bu * vert1.color[2] + bv * vert2.color[2];
				vertexColor[3] = bw * vert0.color[3] + bu * vert1.color[3] + bv * vert2.color[3];

				albedo[0] *= vertexColor[0];
				albedo[1] *= vertexColor[1];
				albedo[2] *= vertexColor[2];
				albedo[3] *= vertexColor[3];
			}

			// ---------------------------------------------------------------
			// Alpha test: skip transparent fragments
			// ---------------------------------------------------------------
			if (material.alphaTest > 0.0f && albedo[3] < material.alphaTest) {
				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // transparent skip — does NOT increment depth
			}

			// nearly invisible alpha blend - skip
			if (material.blendMode == 1 && albedo[3] < 0.05f) {
				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // transparent skip — does NOT increment depth
			}

			// ---------------------------------------------------------------
			// Lit noshadows decal overlays
			// ---------------------------------------------------------------
			if (material.noShadows && !material.isAmbientOnly && material.alphaTest == 0.0f
				&& material.transmission == 0.0f && material.blendMode == 0) {
				float alpha = albedo[3];
				float lum = albedo[0] * 0.2126f + albedo[1] * 0.7152f + albedo[2] * 0.0722f;
				if (lum < 0.02f) alpha = 0.0f;

				if (alpha < 0.05f) {
					ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
					ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
					ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
					ray.tMin = rayOffset;
					continue;
				}

				// light the decal overlay
				if (numLights > 0) {
					int numSamples = min(numLights, maxLightSamples);
					float lightWeight = (float)numLights / (float)numSamples;
					for (int i = 0; i < numSamples; i++) {
						int lightIndex = (int)(curand_uniform(&randState) * numLights);
						if (lightIndex >= numLights) lightIndex = numLights - 1;
						float directLight[3];
						SampleDirectLight(lights[lightIndex], hitInfo.position, surfaceNormal,
							vertices, triangles, bvhNodes, triIndices, materials, textures, &randState, directLight);
						pathRadiance[0] += pathThroughput[0] * directLight[0] * albedo[0] * alpha * lightWeight;
						pathRadiance[1] += pathThroughput[1] * directLight[1] * albedo[1] * alpha * lightWeight;
						pathRadiance[2] += pathThroughput[2] * directLight[2] * albedo[2] * alpha * lightWeight;
					}
				}

				pathRadiance[0] += pathThroughput[0] * material.emission[0] * emissionBoost * alpha;
				pathRadiance[1] += pathThroughput[1] * material.emission[1] * emissionBoost * alpha;
				pathRadiance[2] += pathThroughput[2] * material.emission[2] * emissionBoost * alpha;

				pathThroughput[0] *= (1.0f - alpha);
				pathThroughput[1] *= (1.0f - alpha);
				pathThroughput[2] *= (1.0f - alpha);

				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // decal overlay — does NOT increment depth
			}

			// ---------------------------------------------------------------
			// Ambient-only materials (self-illuminated, HUD, etc)
			// ---------------------------------------------------------------
			if (material.isAmbientOnly) {
				float alpha = albedo[3];

				if (material.blendMode == 2) {
					// additive
					pathRadiance[0] += pathThroughput[0] * albedo[0] * alpha;
					pathRadiance[1] += pathThroughput[1] * albedo[1] * alpha;
					pathRadiance[2] += pathThroughput[2] * albedo[2] * alpha;
				} else if (material.blendMode == 3) {
					// modulate
					pathThroughput[0] *= albedo[0];
					pathThroughput[1] *= albedo[1];
					pathThroughput[2] *= albedo[2];
				} else {
					// default alpha-over
					pathRadiance[0] += pathThroughput[0] * albedo[0] * alpha;
					pathRadiance[1] += pathThroughput[1] * albedo[1] * alpha;
					pathRadiance[2] += pathThroughput[2] * albedo[2] * alpha;
					pathThroughput[0] *= (1.0f - alpha);
					pathThroughput[1] *= (1.0f - alpha);
					pathThroughput[2] *= (1.0f - alpha);
				}

				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // ambient-only overlay — does NOT increment depth
			}

			// ---------------------------------------------------------------
			// Additive blending
			// ---------------------------------------------------------------
			if (material.blendMode == 2) {
				pathRadiance[0] += pathThroughput[0] * albedo[0] * albedo[3];
				pathRadiance[1] += pathThroughput[1] * albedo[1] * albedo[3];
				pathRadiance[2] += pathThroughput[2] * albedo[2] * albedo[3];
				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // additive overlay — does NOT increment depth
			} else if (material.blendMode == 3) {
				// modulate
				pathThroughput[0] *= albedo[0];
				pathThroughput[1] *= albedo[1];
				pathThroughput[2] *= albedo[2];
				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // modulate overlay — does NOT increment depth
			}

			// ---------------------------------------------------------------
			// Transmission / glass: Fresnel reflection vs refraction
			// ---------------------------------------------------------------
			if (material.transmission > 0.0f) {
				float negDir[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
				float cosTheta = fabsf(dot3(surfaceNormal, negDir));
				float r0 = (material.ior - 1.0f) / (material.ior + 1.0f);
				r0 = r0 * r0;
				float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
				float reflectProb = fresnel + (1.0f - fresnel) * (1.0f - material.transmission);

				if (curand_uniform(&randState) > reflectProb) {
					// transmit through
					pathThroughput[0] *= albedo[0];
					pathThroughput[1] *= albedo[1];
					pathThroughput[2] *= albedo[2];
					ray.origin[0] = hitInfo.position[0] - surfaceNormal[0] * rayOffset;
					ray.origin[1] = hitInfo.position[1] - surfaceNormal[1] * rayOffset;
					ray.origin[2] = hitInfo.position[2] - surfaceNormal[2] * rayOffset;
					ray.tMin = rayOffset;
					continue;  // glass transmission — does NOT increment depth
				}
			}

			// ---------------------------------------------------------------
			// Emission
			// ---------------------------------------------------------------
			pathRadiance[0] += pathThroughput[0] * material.emission[0] * emissionBoost;
			pathRadiance[1] += pathThroughput[1] * material.emission[1] * emissionBoost;
			pathRadiance[2] += pathThroughput[2] * material.emission[2] * emissionBoost;

			// ---------------------------------------------------------------
			// Direct light sampling with PBR BRDF
			// ---------------------------------------------------------------
			if (numLights > 0) {
				int numSamples = min(numLights, maxLightSamples);
				float lightWeight = (float)numLights / (float)numSamples;

				for (int i = 0; i < numSamples; i++) {
					int lightIndex = (int)(curand_uniform(&randState) * numLights);
					if (lightIndex >= numLights) lightIndex = numLights - 1;

					float directLight[3];
					SampleDirectLight(lights[lightIndex], hitInfo.position, surfaceNormal,
						vertices, triangles, bvhNodes, triIndices, materials, textures, &randState, directLight);

					if (material.blendMode == 0 || material.blendMode == 1) {
						// compute light direction for PBR BRDF evaluation
						float wi[3];
						if (lights[lightIndex].type == 1) {
							wi[0] = -lights[lightIndex].direction[0];
							wi[1] = -lights[lightIndex].direction[1];
							wi[2] = -lights[lightIndex].direction[2];
						} else {
							wi[0] = lights[lightIndex].position[0] - hitInfo.position[0];
							wi[1] = lights[lightIndex].position[1] - hitInfo.position[1];
							wi[2] = lights[lightIndex].position[2] - hitInfo.position[2];
						}
						normalize3(wi);

						float wo[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
						float directBrdf[3];
						EvaluatePBRMaterial(albedo, surfaceNormal, wo, wi, metallic, roughness, specularBoost, specColor, directBrdf);

						float alphaScale = (material.blendMode == 1) ? albedo[3] : 1.0f;
						pathRadiance[0] += pathThroughput[0] * directLight[0] * directBrdf[0] * alphaScale * lightWeight;
						pathRadiance[1] += pathThroughput[1] * directLight[1] * directBrdf[1] * alphaScale * lightWeight;
						pathRadiance[2] += pathThroughput[2] * directLight[2] * directBrdf[2] * alphaScale * lightWeight;
					} else {
						// fallback Lambertian
						pathRadiance[0] += pathThroughput[0] * directLight[0] * albedo[0] * lightWeight;
						pathRadiance[1] += pathThroughput[1] * directLight[1] * albedo[1] * lightWeight;
						pathRadiance[2] += pathThroughput[2] * directLight[2] * albedo[2] * lightWeight;
					}

					// -------------------------------------------------------
					// Volumetric scattering
					// -------------------------------------------------------
					if (depth < 2 && lights[lightIndex].volumetric > 0.0f) {
						float volumetric[3];
						float rayLen = hitInfo.t;
						float rayDir[3] = { ray.direction[0], ray.direction[1], ray.direction[2] };
						float rayOrg[3] = { ray.origin[0], ray.origin[1], ray.origin[2] };

						SampleVolumetricScattering(lights[lightIndex], rayOrg, rayDir, rayLen,
							vertices, triangles, bvhNodes, triIndices, materials, textures, &randState, volumetric,
							volumetricDensity, volumetricSteps, volumetricAnisotropy, volFalloff, volMaxDist);

						pathRadiance[0] += pathThroughput[0] * volumetric[0] * lightWeight;
						pathRadiance[1] += pathThroughput[1] * volumetric[1] * lightWeight;
						pathRadiance[2] += pathThroughput[2] * volumetric[2] * lightWeight;
					}
				}
			}

			// No ambient fill — Doom 3 scenes are lit exclusively by placed lights.

			// ---------------------------------------------------------------
			// Indirect lighting (one-bounce GI)
			// ---------------------------------------------------------------
			if (depth == 0 && indirectProb > 0.0f && curand_uniform(&randState) < indirectProb) {
				float indirectDir[3];
				float indirectPdf;
				SampleHemisphere(surfaceNormal, &randState, indirectDir, indirectPdf);

				if (indirectPdf > 1e-6f) {
					float indirectLight[3];
					SampleIndirectLight(hitInfo.position, surfaceNormal, indirectDir,
						vertices, triangles, bvhNodes, triIndices, materials, textures,
						lights, numLights, &randState, indirectLight);

					if (!isfinite(indirectLight[0])) indirectLight[0] = 0.0f;
					if (!isfinite(indirectLight[1])) indirectLight[1] = 0.0f;
					if (!isfinite(indirectLight[2])) indirectLight[2] = 0.0f;

					if (material.blendMode == 0 || material.blendMode == 1) {
						float indirectBrdf[3];
						float wo[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
						float NdotL_indirect = fmaxf(0.0f, dot3(surfaceNormal, indirectDir));

						EvaluatePBRMaterial(albedo, surfaceNormal, wo, indirectDir, metallic, roughness, specularBoost, specColor, indirectBrdf);

						float indirectScale = NdotL_indirect / (indirectProb * indirectPdf);
						if (indirectScale > 10.0f) indirectScale = 10.0f;
						float alphaScale = (material.blendMode == 1) ? albedo[3] : 1.0f;
						pathRadiance[0] += pathThroughput[0] * indirectLight[0] * indirectBrdf[0] * indirectScale * alphaScale;
						pathRadiance[1] += pathThroughput[1] * indirectLight[1] * indirectBrdf[1] * indirectScale * alphaScale;
						pathRadiance[2] += pathThroughput[2] * indirectLight[2] * indirectBrdf[2] * indirectScale * alphaScale;
					} else {
						float indirectBrdf[3];
						float wo[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
						EvaluateMaterial(albedo, surfaceNormal, wo, indirectDir, indirectBrdf);

						float indirectScale = (1.0f / indirectProb) / indirectPdf;
						if (indirectScale > 10.0f) indirectScale = 10.0f;
						pathRadiance[0] += pathThroughput[0] * indirectLight[0] * indirectBrdf[0] * indirectScale;
						pathRadiance[1] += pathThroughput[1] * indirectLight[1] * indirectBrdf[1] * indirectScale;
						pathRadiance[2] += pathThroughput[2] * indirectLight[2] * indirectBrdf[2] * indirectScale;
					}
				}
			}

			// ---------------------------------------------------------------
			// Throughput early termination
			// ---------------------------------------------------------------
			if (throughputClamp > 0.0f && pathThroughput[0] < throughputClamp && pathThroughput[1] < throughputClamp && pathThroughput[2] < throughputClamp) {
				break;
			}

			// alpha blend pass-through for semi-transparent surfaces
			if (material.blendMode == 1 && albedo[3] < 0.95f) {
				float alpha = albedo[3];
				pathThroughput[0] *= (1.0f - alpha);
				pathThroughput[1] *= (1.0f - alpha);
				pathThroughput[2] *= (1.0f - alpha);
				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * rayOffset;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * rayOffset;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * rayOffset;
				ray.tMin = rayOffset;
				continue;  // alpha blend pass-through — does NOT increment depth
			}

			// only real opaque/PBR bounces consume depth budget
			depth++;

			// ---------------------------------------------------------------
			// Russian roulette
			// ---------------------------------------------------------------
			if (rrEnabled && depth >= rrMinBounces) {
				float maxThroughput = fmaxf(pathThroughput[0], fmaxf(pathThroughput[1], pathThroughput[2]));
				float survivalProb = fminf(0.95f, fmaxf(rrSurvivalMin, maxThroughput));
				if (curand_uniform(&randState) > survivalProb) {
					break;
				}
				float invProb = 1.0f / survivalProb;
				pathThroughput[0] *= invProb;
				pathThroughput[1] *= invProb;
				pathThroughput[2] *= invProb;
			}

			// early termination
			if (earlyTermThreshold > 0.0f) {
				float maxThroughput = fmaxf(pathThroughput[0], fmaxf(pathThroughput[1], pathThroughput[2]));
				if (maxThroughput < earlyTermThreshold) break;
			}

			// ---------------------------------------------------------------
			// Sample next bounce direction (cosine-weighted hemisphere)
			// ---------------------------------------------------------------
			float nextDir[3];
			float pdf;
			SampleHemisphere(surfaceNormal, &randState, nextDir, pdf);

			if (pdf < 1e-6f) break;

			// update throughput with PBR BRDF
			float brdf[3];
			float wo[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
			float NdotL_bounce = fmaxf(0.0f, dot3(surfaceNormal, nextDir));

			if (material.blendMode == 0 || material.blendMode == 1) {
				EvaluatePBRMaterial(albedo, surfaceNormal, wo, nextDir, metallic, roughness, specularBoost, specColor, brdf);
			} else {
				EvaluateMaterial(albedo, surfaceNormal, wo, nextDir, brdf);
			}

			pathThroughput[0] *= brdf[0] * NdotL_bounce / pdf;
			pathThroughput[1] *= brdf[1] * NdotL_bounce / pdf;
			pathThroughput[2] *= brdf[2] * NdotL_bounce / pdf;

			// NaN/Inf guard on throughput
			if (!isfinite(pathThroughput[0]) || !isfinite(pathThroughput[1]) || !isfinite(pathThroughput[2])) {
				break;
			}

			// next ray from surface
			ray.origin[0] = hitInfo.position[0] + surfaceNormal[0] * rayOffset;
			ray.origin[1] = hitInfo.position[1] + surfaceNormal[1] * rayOffset;
			ray.origin[2] = hitInfo.position[2] + surfaceNormal[2] * rayOffset;
			ray.direction[0] = nextDir[0];
			ray.direction[1] = nextDir[1];
			ray.direction[2] = nextDir[2];
			ray.tMin = rayOffset;
			ray.tMax = 1e10f;

		}  // end bounce loop

		color[0] += pathRadiance[0];
		color[1] += pathRadiance[1];
		color[2] += pathRadiance[2];

	}  // end sample loop

	// average over samples
	color[0] /= (float)samplesPerPixel;
	color[1] /= (float)samplesPerPixel;
	color[2] /= (float)samplesPerPixel;

	// firefly clamp
	if (fireflyClamp > 0.0f) {
		color[0] = fminf(color[0], fireflyClamp);
		color[1] = fminf(color[1], fireflyClamp);
		color[2] = fminf(color[2], fireflyClamp);
	}

	// NaN/Inf guard
	if (!isfinite(color[0]) || !isfinite(color[1]) || !isfinite(color[2])) {
		color[0] = color[1] = color[2] = 0.0f;
	}

	// additive temporal accumulation
	framebuffer[pixelIndex + 0] += color[0];
	framebuffer[pixelIndex + 1] += color[1];
	framebuffer[pixelIndex + 2] += color[2];
	framebuffer[pixelIndex + 3] = 1.0f;
}

/*
=================================================================================
ToneMappingKernel - Converts HDR accumulated buffer to LDR output
=================================================================================
*/
__global__ void ToneMappingKernel(
	const float* hdrBuffer,
	unsigned char* ldrBuffer,
	int width,
	int height,
	float exposure,
	float gamma,
	int toneMapMode,
	int frameCount
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = (y * width + x) * 4;

	float invFrameCount = 1.0f / fmaxf(1.0f, (float)frameCount);
	float r = hdrBuffer[pixelIndex + 0] * invFrameCount;
	float g = hdrBuffer[pixelIndex + 1] * invFrameCount;
	float b = hdrBuffer[pixelIndex + 2] * invFrameCount;

	// apply exposure
	r *= exposure;
	g *= exposure;
	b *= exposure;

	// tone mapping
	if (toneMapMode == 1) {
		// ACES approximation
		float a = 2.51f;
		float bp = 0.03f;
		float c = 2.43f;
		float d = 0.59f;
		float e = 0.14f;
		r = (r * (a * r + bp)) / (r * (c * r + d) + e);
		g = (g * (a * g + bp)) / (g * (c * g + d) + e);
		b = (b * (a * b + bp)) / (b * (c * b + d) + e);
	} else if (toneMapMode == 2) {
		// uncharted 2
		float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
		float W = 11.2f;
		float mapW = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
		float whiteScale = 1.0f / mapW;
		r = (((r * (A * r + C * B) + D * E) / (r * (A * r + B) + D * F)) - E / F) * whiteScale;
		g = (((g * (A * g + C * B) + D * E) / (g * (A * g + B) + D * F)) - E / F) * whiteScale;
		b = (((b * (A * b + C * B) + D * E) / (b * (A * b + B) + D * F)) - E / F) * whiteScale;
	} else {
		// reinhard (default, mode 0)
		r = r / (1.0f + r);
		g = g / (1.0f + g);
		b = b / (1.0f + b);
	}

	// gamma correction
	float invGamma = 1.0f / gamma;
	r = powf(fmaxf(r, 0.0f), invGamma);
	g = powf(fmaxf(g, 0.0f), invGamma);
	b = powf(fmaxf(b, 0.0f), invGamma);

	// convert to 8-bit LDR
	ldrBuffer[pixelIndex + 0] = (unsigned char)(clamp(r, 0.0f, 1.0f) * 255.0f);
	ldrBuffer[pixelIndex + 1] = (unsigned char)(clamp(g, 0.0f, 1.0f) * 255.0f);
	ldrBuffer[pixelIndex + 2] = (unsigned char)(clamp(b, 0.0f, 1.0f) * 255.0f);
	ldrBuffer[pixelIndex + 3] = 255;
}
