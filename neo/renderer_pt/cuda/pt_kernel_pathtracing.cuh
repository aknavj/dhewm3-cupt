#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"
#include "renderer_pt/cuda/pt_cuda_material.cuh"

#define PT_BVH_STACK	128	// LBVH trees are not depth bounded

/*
=================================================================================
IntersectTriangleUV

Moller-Trumbore, returns the ray distance and barycentrics only. The hit
attributes are interpolated once for the closest hit by PT_FillHitAttributes.
=================================================================================
*/
__device__ __forceinline__ bool IntersectTriangleUV(
	const float* orig,
	const float* dir,
	const ptVertex_t* vertices,
	const ptTriangle_t& tri,
	float& t,
	float& u,
	float& v
) {
	const float* p0 = vertices[tri.vertexIndices[0]].position;
	const float* p1 = vertices[tri.vertexIndices[1]].position;
	const float* p2 = vertices[tri.vertexIndices[2]].position;

	const float e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
	const float e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };

	float h[3];
	cross3(dir, e2, h);
	const float a = dot3(e1, h);
	if (fabsf(a) < 1e-8f) return false;

	const float f = 1.0f / a;
	const float s[3] = { orig[0] - p0[0], orig[1] - p0[1], orig[2] - p0[2] };
	u = f * dot3(s, h);
	if (u < 0.0f || u > 1.0f) return false;

	float q[3];
	cross3(s, e1, q);
	v = f * dot3(dir, q);
	if (v < 0.0f || u + v > 1.0f) return false;

	t = f * dot3(e2, q);
	return true;
}

/*
=================================================================================
PT_FillHitAttributes
=================================================================================
*/
__device__ void PT_FillHitAttributes(
	const ptRay_t& ray,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	int triIdx,
	float t,
	float u,
	float v,
	ptHitInfo_t& hitInfo
) {
	const ptTriangle_t& tri = triangles[triIdx];
	const ptVertex_t& v0 = vertices[tri.vertexIndices[0]];
	const ptVertex_t& v1 = vertices[tri.vertexIndices[1]];
	const ptVertex_t& v2 = vertices[tri.vertexIndices[2]];
	const float w = 1.0f - u - v;

	hitInfo.hit = true;
	hitInfo.t = t;
	for (int i = 0; i < 3; i++) {
		hitInfo.position[i] = ray.origin[i] + t * ray.direction[i];
		hitInfo.normal[i] = w * v0.normal[i] + u * v1.normal[i] + v * v2.normal[i];
		hitInfo.tangent[i] = w * v0.tangent[i] + u * v1.tangent[i] + v * v2.tangent[i];
		hitInfo.binormal[i] = w * v0.binormal[i] + u * v1.binormal[i] + v * v2.binormal[i];
	}
	normalize3(hitInfo.normal);
	normalize3(hitInfo.tangent);
	normalize3(hitInfo.binormal);

	hitInfo.texcoord[0] = w * v0.texcoord[0] + u * v1.texcoord[0] + v * v2.texcoord[0];
	hitInfo.texcoord[1] = w * v0.texcoord[1] + u * v1.texcoord[1] + v * v2.texcoord[1];

	hitInfo.barycentrics[0] = w;
	hitInfo.barycentrics[1] = u;
	hitInfo.barycentrics[2] = v;

	hitInfo.materialIndex = tri.materialIndex;
	hitInfo.triangleIndex = triIdx;
	hitInfo.triFlags = tri.flags;
}

/*
=================================================================================
Slab test helpers
=================================================================================
*/
__device__ __forceinline__ void PT_InvDir(const float* dir, float* invD) {
	for (int i = 0; i < 3; i++) {
		// a huge finite value keeps (bound - origin) * invD free of 0 * inf NaNs
		invD[i] = fabsf(dir[i]) > 1e-12f ? 1.0f / dir[i] : copysignf(1e30f, dir[i]);
	}
}

__device__ __forceinline__ bool PT_RayBox(const float* orig, const float* invD, const float* b, float tmin, float tmax) {
	for (int i = 0; i < 3; i++) {
		const float t0 = (b[i] - orig[i]) * invD[i];
		const float t1 = (b[i + 3] - orig[i]) * invD[i];
		tmin = fmaxf(tmin, fminf(t0, t1));
		tmax = fminf(tmax, fmaxf(t0, t1));
	}
	return tmin <= tmax;
}

// pushes the children of an inner node, the one closer along the ray on top
__device__ __forceinline__ void PT_PushChildren(const ptBVHNode_t& node, const ptBVHNode_t* nodes, const float* orig,
												const float* dir, int* stack, int& stackPtr) {
	if (stackPtr > PT_BVH_STACK - 2) {
		return;
	}
	const float* lb = nodes[node.leftChild].bounds;
	const float* rb = nodes[node.rightChild].bounds;
	float leftDot = 0.0f, rightDot = 0.0f;
	for (int i = 0; i < 3; i++) {
		leftDot += ((lb[i] + lb[i + 3]) - 2.0f * orig[i]) * dir[i];
		rightDot += ((rb[i] + rb[i + 3]) - 2.0f * orig[i]) * dir[i];
	}
	if (leftDot < rightDot) {
		stack[stackPtr++] = node.rightChild;
		stack[stackPtr++] = node.leftChild;
	} else {
		stack[stackPtr++] = node.leftChild;
		stack[stackPtr++] = node.rightChild;
	}
}

/*
=================================================================================
IntersectBVH - closest hit

View weapon handling (the GL renderer draws it with a depth hack, so its
geometry sits inside walls and floors): PT_TRI_WEAPON triangles are only
visible to rays with PT_RAY_SEES_WEAPON and then win over any world hit
within PT_WEAPON_REACH. World triangles closer than ray.worldTMin are
ignored, which lets rays leaving the weapon escape the geometry it clips.
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
	float invD[3];
	PT_InvDir(ray.direction, invD);

	const bool seesWeapon = (ray.flags & PT_RAY_SEES_WEAPON) != 0;
	const float worldMin = fmaxf(ray.tMin, ray.worldTMin);
	const float weaponMax = fminf(ray.tMax, PT_WEAPON_REACH);

	float bestT = ray.tMax;
	float bestU = 0.0f, bestV = 0.0f;
	int bestTri = -1;
	bool bestWeapon = false;

	int stack[PT_BVH_STACK];
	int stackPtr = 0;
	stack[stackPtr++] = 0;

	while (stackPtr > 0) {
		const ptBVHNode_t& node = nodes[stack[--stackPtr]];

		const float cullT = (seesWeapon && !bestWeapon) ? fmaxf(bestT, weaponMax) : bestT;
		if (!PT_RayBox(ray.origin, invD, node.bounds, ray.tMin, cullT)) {
			continue;
		}

		if (node.leftChild != -1) {
			PT_PushChildren(node, nodes, ray.origin, ray.direction, stack, stackPtr);
			continue;
		}

		for (int i = node.firstPrimitive; i < node.firstPrimitive + node.primitiveCount; i++) {
			const int triIdx = triIndices[i];
			const ptTriangle_t& tri = triangles[triIdx];
			const bool isWeapon = (tri.flags & PT_TRI_WEAPON) != 0;
			if (isWeapon ? !seesWeapon : bestWeapon) {
				continue;
			}

			float t, u, v;
			if (!IntersectTriangleUV(ray.origin, ray.direction, vertices, tri, t, u, v)) {
				continue;
			}

			if (isWeapon) {
				if (t <= ray.tMin || t >= weaponMax || (bestWeapon && t >= bestT)) continue;
				bestWeapon = true;
			} else if (t <= worldMin || t >= bestT) {
				continue;
			}
			bestT = t;
			bestU = u;
			bestV = v;
			bestTri = triIdx;
		}
	}

	if (bestTri < 0) {
		return false;
	}
	PT_FillHitAttributes(ray, vertices, triangles, bestTri, bestT, bestU, bestV, hitInfo);
	return true;
}

/*
=================================================================================
PT_ApplyTexTransform
=================================================================================
*/
__device__ __forceinline__ void PT_ApplyTexTransform(const ptMaterial_t& material, float& tu, float& tv) {
	const float* xf = material.texTransform;
	if (xf[0] == 1.0f && xf[1] == 1.0f && xf[2] == 0.0f && xf[3] == 0.0f && xf[4] == 0.0f) {
		return;
	}
	tu *= xf[0];
	tv *= xf[1];
	if (xf[2] != 0.0f) {
		float sinA, cosA;
		sincosf(xf[2], &sinA, &cosA);
		const float uRot = tu * cosA - tv * sinA;
		const float vRot = tu * sinA + tv * cosA;
		tu = uRot;
		tv = vRot;
	}
	tu += xf[3];
	tv += xf[4];
	tu = tu - floorf(tu);
	tv = tv - floorf(tv);
}

/*
=================================================================================
PT_ShadowTransmit

Filters a shadow ray through a non-opaque surface. Returns false when the
surface blocks the ray.
=================================================================================
*/
__device__ bool PT_ShadowTransmit(
	const float* dir,
	const ptVertex_t* vertices,
	const ptTriangle_t& tri,
	float u,
	float v,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	float* atten
) {
	const ptMaterial_t& m = materials[tri.materialIndex];
	if (m.noShadows || m.isAmbientOnly || m.blendMode == PT_BLEND_ADD) {
		return true;
	}

	float tex[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	if (m.albedoTexture >= 0 && textures) {
		const ptVertex_t& v0 = vertices[tri.vertexIndices[0]];
		const ptVertex_t& v1 = vertices[tri.vertexIndices[1]];
		const ptVertex_t& v2 = vertices[tri.vertexIndices[2]];
		const float w = 1.0f - u - v;
		float tu = w * v0.texcoord[0] + u * v1.texcoord[0] + v * v2.texcoord[0];
		float tv = w * v0.texcoord[1] + u * v1.texcoord[1] + v * v2.texcoord[1];
		PT_ApplyTexTransform(m, tu, tv);
		SampleTextureSrgb(textures, m.albedoTexture, tu, tv, tex);
	}
	const float alpha = m.albedo[3] * tex[3];

	if (m.alphaTest > 0.0f && alpha < m.alphaTest) {
		return true;
	}

	if (m.transmission > 0.0f) {
		const float* p0 = vertices[tri.vertexIndices[0]].position;
		const float* p1 = vertices[tri.vertexIndices[1]].position;
		const float* p2 = vertices[tri.vertexIndices[2]].position;
		const float e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
		const float e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
		float n[3];
		cross3(e1, e2, n);
		normalize3(n);

		const float cosT = fabsf(dot3(n, dir));
		float r0 = (m.ior - 1.0f) / (m.ior + 1.0f);
		r0 = r0 * r0;
		const float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosT, 5.0f);
		const float transmitFrac = (1.0f - fresnel) * m.transmission;
		for (int c = 0; c < 3; c++) {
			atten[c] *= transmitFrac * m.albedo[c] * tex[c];
		}
	} else if (m.blendMode == PT_BLEND_ALPHA) {
		const float transmit = 1.0f - fminf(alpha, 1.0f);
		atten[0] *= transmit;
		atten[1] *= transmit;
		atten[2] *= transmit;
	} else {
		return false;
	}

	return atten[0] >= 0.01f || atten[1] >= 0.01f || atten[2] >= 0.01f;
}

/*
=================================================================================
TraceShadow - any hit visibility with colored transmission

Returns false when the segment [max(tMin, worldTMin), tMax] is blocked,
otherwise atten holds the product of the transmittance of every
translucent surface along it. Weapon triangles never cast shadows.
=================================================================================
*/
__device__ bool TraceShadow(
	const ptRay_t& ray,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* nodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	float* atten
) {
	atten[0] = atten[1] = atten[2] = 1.0f;

	float invD[3];
	PT_InvDir(ray.direction, invD);
	const float tMin = fmaxf(ray.tMin, ray.worldTMin);

	int stack[PT_BVH_STACK];
	int stackPtr = 0;
	stack[stackPtr++] = 0;

	while (stackPtr > 0) {
		const ptBVHNode_t& node = nodes[stack[--stackPtr]];
		if (!PT_RayBox(ray.origin, invD, node.bounds, tMin, ray.tMax)) {
			continue;
		}

		if (node.leftChild != -1) {
			PT_PushChildren(node, nodes, ray.origin, ray.direction, stack, stackPtr);
			continue;
		}

		for (int i = node.firstPrimitive; i < node.firstPrimitive + node.primitiveCount; i++) {
			const ptTriangle_t& tri = triangles[triIndices[i]];
			if (tri.flags & (PT_TRI_WEAPON | PT_TRI_NO_OCCLUDE)) {
				continue;
			}

			float t, u, v;
			if (!IntersectTriangleUV(ray.origin, ray.direction, vertices, tri, t, u, v) || t <= tMin || t >= ray.tMax) {
				continue;
			}
			if (tri.flags & PT_TRI_OPAQUE) {
				return false;
			}
			if (!PT_ShadowTransmit(ray.direction, vertices, tri, u, v, materials, textures, atten)) {
				return false;
			}
		}
	}

	return true;
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

	ptRng_t rng;
	PT_RngInit(&rng, (unsigned int)(y * width + x), (unsigned int)frameIndex);

	float color[3] = { 0.0f, 0.0f, 0.0f };

	for (int sample = 0; sample < samplesPerPixel; sample++) {
		// primary ray with jitter
		float jitterX = PT_Rand(&rng) - 0.5f;
		float jitterY = PT_Rand(&rng) - 0.5f;

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
		ray.worldTMin = 0.0f;
		ray.flags = PT_RAY_SEES_WEAPON;

		float pathThroughput[3] = { 1.0f, 1.0f, 1.0f };
		float pathRadiance[3] = { 0.0f, 0.0f, 0.0f };
		float lastShadePos[3] = { 0.0f, 0.0f, 0.0f };	// where the previous bounce did light sampling

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

			// pass-through continuations restart at the hit, keep the clearance where it was
			ray.worldTMin = fmaxf(0.0f, ray.worldTMin - hitInfo.t);

			const ptMaterial_t& material = materials[hitInfo.materialIndex];

			// rays leaving the view weapon skip the world geometry it pokes into
			const bool hitWeapon = (hitInfo.triFlags & PT_TRI_WEAPON) != 0;
			const float shadeWorldTMin = hitWeapon ? PT_WEAPON_CLEARANCE : 0.0f;

			// ---------------------------------------------------------------
			// Apply texture transforms to UV
			// ---------------------------------------------------------------
			float tu = hitInfo.texcoord[0];
			float tv = hitInfo.texcoord[1];

			PT_ApplyTexTransform(material, tu, tv);

			// ---------------------------------------------------------------
			// Sample albedo texture
			// ---------------------------------------------------------------
			float albedo[4];
			if (material.albedoTexture >= 0 && textures) {
				SampleTextureSrgb(textures, material.albedoTexture, tu, tv, albedo);
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

				albedo[0] *= PT_SrgbToLinear(vertexColor[0]);
				albedo[1] *= PT_SrgbToLinear(vertexColor[1]);
				albedo[2] *= PT_SrgbToLinear(vertexColor[2]);
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
				int picks[PT_MAX_LIGHT_PICKS];
				float pickWeights[PT_MAX_LIGHT_PICKS];
				const int numPicks = PT_SelectLights(lights, numLights, hitInfo.position, surfaceNormal,
					min(numLights, maxLightSamples), &rng, picks, pickWeights);
				for (int i = 0; i < numPicks; i++) {
					float directLight[3], wi[3];
					SampleDirectLight(lights[picks[i]], hitInfo.position, surfaceNormal, shadeWorldTMin,
						vertices, triangles, bvhNodes, triIndices, materials, textures, &rng, directLight, wi);
					const float w = alpha * pickWeights[i];
					pathRadiance[0] += pathThroughput[0] * directLight[0] * albedo[0] * w;
					pathRadiance[1] += pathThroughput[1] * directLight[1] * albedo[1] * w;
					pathRadiance[2] += pathThroughput[2] * directLight[2] * albedo[2] * w;
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

				if (PT_Rand(&rng) > reflectProb) {
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
			// skip emitters the previous bounce's light sampling already covered
			if (depth == 0 || !(hitInfo.triFlags & PT_TRI_AREA_LIGHT)
				|| !PT_AreaLightSampledFrom(vertices, triangles[hitInfo.triangleIndex], lastShadePos)) {
				float glow[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
				if (material.emissionTexture >= 0 && textures) {
					SampleTextureSrgb(textures, material.emissionTexture, tu, tv, glow);
				}
				pathRadiance[0] += pathThroughput[0] * material.emission[0] * glow[0] * emissionBoost;
				pathRadiance[1] += pathThroughput[1] * material.emission[1] * glow[1] * emissionBoost;
				pathRadiance[2] += pathThroughput[2] * material.emission[2] * glow[2] * emissionBoost;
			}

			// ---------------------------------------------------------------
			// Direct light sampling with PBR BRDF
			// ---------------------------------------------------------------
			{
				int picks[PT_MAX_LIGHT_PICKS];
				float pickWeights[PT_MAX_LIGHT_PICKS];
				const int numPicks = PT_SelectLights(lights, numLights, hitInfo.position, surfaceNormal,
					min(numLights, maxLightSamples), &rng, picks, pickWeights);

				for (int i = 0; i < numPicks; i++) {
					float directLight[3], wi[3];
					SampleDirectLight(lights[picks[i]], hitInfo.position, surfaceNormal, shadeWorldTMin,
						vertices, triangles, bvhNodes, triIndices, materials, textures, &rng, directLight, wi);
					if (directLight[0] + directLight[1] + directLight[2] <= 0.0f) {
						continue;
					}

					const float lightWeight = pickWeights[i];
					if (material.blendMode == PT_BLEND_OPAQUE || material.blendMode == PT_BLEND_ALPHA) {
						float wo[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
						float directBrdf[3];
						EvaluatePBRMaterial(albedo, surfaceNormal, wo, wi, metallic, roughness, specularBoost, specColor, directBrdf);

						float alphaScale = (material.blendMode == PT_BLEND_ALPHA) ? albedo[3] : 1.0f;
						pathRadiance[0] += pathThroughput[0] * directLight[0] * directBrdf[0] * alphaScale * lightWeight;
						pathRadiance[1] += pathThroughput[1] * directLight[1] * directBrdf[1] * alphaScale * lightWeight;
						pathRadiance[2] += pathThroughput[2] * directLight[2] * directBrdf[2] * alphaScale * lightWeight;
					} else {
						// fallback Lambertian
						pathRadiance[0] += pathThroughput[0] * directLight[0] * albedo[0] * lightWeight;
						pathRadiance[1] += pathThroughput[1] * directLight[1] * albedo[1] * lightWeight;
						pathRadiance[2] += pathThroughput[2] * directLight[2] * albedo[2] * lightWeight;
					}
				}
			}

			// ---------------------------------------------------------------
			// Volumetric scattering along the segment that reached this hit.
			// Uniform light picks: importance at the surface says nothing
			// about the lights the segment passes through.
			// ---------------------------------------------------------------
			if (depth < 2 && volumetricDensity > 0.0f && numLights > 0) {
				const int numSamples = min(numLights, maxLightSamples);
				const float lightWeight = (float)numLights / (float)numSamples;
				for (int i = 0; i < numSamples; i++) {
					const int lightIndex = min((int)(PT_Rand(&rng) * numLights), numLights - 1);
					if (lights[lightIndex].volumetric <= 0.0f) {
						continue;
					}
					float volumetric[3];
					SampleVolumetricScattering(lights[lightIndex], ray.origin, ray.direction, hitInfo.t,
						vertices, triangles, bvhNodes, triIndices, materials, textures, &rng, volumetric,
						volumetricDensity, volumetricSteps, volumetricAnisotropy, volFalloff, volMaxDist);

					pathRadiance[0] += pathThroughput[0] * volumetric[0] * lightWeight;
					pathRadiance[1] += pathThroughput[1] * volumetric[1] * lightWeight;
					pathRadiance[2] += pathThroughput[2] * volumetric[2] * lightWeight;
				}
			}

			// No ambient fill — Doom 3 scenes are lit exclusively by placed lights.

			// ---------------------------------------------------------------
			// Indirect lighting (one-bounce GI), only for single bounce
			// paths: deeper paths already continue into the same bounce
			// ---------------------------------------------------------------
			if (depth == 0 && maxDepth <= 1 && indirectProb > 0.0f && PT_Rand(&rng) < indirectProb) {
				float indirectDir[3];
				float indirectPdf;
				SampleHemisphere(surfaceNormal, &rng, indirectDir, indirectPdf);

				if (indirectPdf > 1e-6f) {
					float indirectLight[3];
					SampleIndirectLight(hitInfo.position, surfaceNormal, indirectDir, shadeWorldTMin,
						vertices, triangles, bvhNodes, triIndices, materials, textures,
						lights, numLights, &rng, indirectLight);

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
			// Low throughput roulette (unbiased replacement for a hard cut)
			// ---------------------------------------------------------------
			if (throughputClamp > 0.0f) {
				const float maxThroughput = fmaxf(pathThroughput[0], fmaxf(pathThroughput[1], pathThroughput[2]));
				if (maxThroughput < throughputClamp) {
					const float survival = maxThroughput / throughputClamp;
					if (PT_Rand(&rng) >= survival) break;
					pathThroughput[0] /= survival;
					pathThroughput[1] /= survival;
					pathThroughput[2] /= survival;
				}
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
				if (PT_Rand(&rng) > survivalProb) {
					break;
				}
				float invProb = 1.0f / survivalProb;
				pathThroughput[0] *= invProb;
				pathThroughput[1] *= invProb;
				pathThroughput[2] *= invProb;
			}

			// early termination roulette
			if (earlyTermThreshold > 0.0f) {
				const float maxThroughput = fmaxf(pathThroughput[0], fmaxf(pathThroughput[1], pathThroughput[2]));
				if (maxThroughput < earlyTermThreshold) {
					const float survival = maxThroughput / earlyTermThreshold;
					if (PT_Rand(&rng) >= survival) break;
					pathThroughput[0] /= survival;
					pathThroughput[1] /= survival;
					pathThroughput[2] /= survival;
				}
			}

			// ---------------------------------------------------------------
			// Sample next bounce direction (cosine-weighted hemisphere)
			// ---------------------------------------------------------------
			float nextDir[3];
			float pdf;
			SampleHemisphere(surfaceNormal, &rng, nextDir, pdf);

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

			lastShadePos[0] = hitInfo.position[0];
			lastShadePos[1] = hitInfo.position[1];
			lastShadePos[2] = hitInfo.position[2];

			// next ray from surface
			ray.origin[0] = hitInfo.position[0] + surfaceNormal[0] * rayOffset;
			ray.origin[1] = hitInfo.position[1] + surfaceNormal[1] * rayOffset;
			ray.origin[2] = hitInfo.position[2] + surfaceNormal[2] * rayOffset;
			ray.direction[0] = nextDir[0];
			ray.direction[1] = nextDir[1];
			ray.direction[2] = nextDir[2];
			ray.tMin = rayOffset;
			ray.tMax = 1e10f;
			ray.worldTMin = shadeWorldTMin;
			ray.flags = 0;

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
