#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"
#include "renderer_pt/cuda/pt_cuda_material.cuh"

/*
========================
TraceShadowRay_RT
BVH shadow ray test
========================
*/
__device__ __forceinline__ bool TraceShadowRay_RT(
	float oX, float oY, float oZ,
	float dX, float dY, float dZ,
	float maxDist,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const int* triIndices,
	const ptBVHNode_t* bvhNodes
) {
	float idX = 1.0f / (fabsf(dX) > 1e-8f ? dX : copysignf(1e-8f, dX));
	float idY = 1.0f / (fabsf(dY) > 1e-8f ? dY : copysignf(1e-8f, dY));
	float idZ = 1.0f / (fabsf(dZ) > 1e-8f ? dZ : copysignf(1e-8f, dZ));

	for (int ni = 0; ni >= 0; ) {
		const ptBVHNode_t& nd = bvhNodes[ni];

		if (!IntersectAABB_Simple(oX, oY, oZ, idX, idY, idZ, nd.bounds, maxDist)) {
			ni = nd.escapeIndex;
			continue;
		}

		if (nd.leftChild == -1) {
			// Leaf
			for (int i = 0; i < nd.primitiveCount; i++) {
				int ti = triIndices[nd.firstPrimitive + i];
				float t = IntersectTriangle_Simple(oX, oY, oZ, dX, dY, dZ,
												   vertices, triangles[ti]);
				if (t > 0.001f && t < maxDist) {
					return true;
				}
			}
			ni = nd.escapeIndex;
		} else {
			ni = nd.leftChild;
		}
	}
	return false;
}

/*
========================
TraceRay_RT
BVH closest-hit traversal with barycentric output
========================
*/
__device__ __forceinline__ float TraceRay_RT(
	float oX, float oY, float oZ,
	float dX, float dY, float dZ,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const int* triIndices,
	const ptBVHNode_t* bvhNodes,
	int& outTriIdx,
	float& outU,
	float& outV
) {
	float idX = 1.0f / (fabsf(dX) > 1e-8f ? dX : copysignf(1e-8f, dX));
	float idY = 1.0f / (fabsf(dY) > 1e-8f ? dY : copysignf(1e-8f, dY));
	float idZ = 1.0f / (fabsf(dZ) > 1e-8f ? dZ : copysignf(1e-8f, dZ));

	float nearest = 1e30f;
	outTriIdx = -1;

	for (int ni = 0; ni >= 0; ) {
		const ptBVHNode_t& nd = bvhNodes[ni];

		if (!IntersectAABB_Simple(oX, oY, oZ, idX, idY, idZ, nd.bounds, nearest)) {
			ni = nd.escapeIndex;
			continue;
		}

		if (nd.leftChild == -1) {
			// leaf
			for (int i = 0; i < nd.primitiveCount; i++) {
				int ti = triIndices[nd.firstPrimitive + i];
				float bu, bv;
				float t = IntersectTriangleUV_Simple(oX, oY, oZ, dX, dY, dZ,
													 vertices, triangles[ti], bu, bv);
				if (t > 0.0f && t < nearest) {
					nearest = t;
					outTriIdx = ti;
					outU = bu;
					outV = bv;
				}
			}
			ni = nd.escapeIndex;
		} else {
			ni = nd.leftChild;
		}
	}
	return nearest;
}

/*
========================
TriangleDrawRayTracedKernel
========================
*/
__global__ void TriangleDrawRayTracedKernel(
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

	float su = (2.0f * ((float)x + 0.5f) / (float)width  - 1.0f) * halfTanX;
	float sv = (2.0f * ((float)y + 0.5f) / (float)height - 1.0f) * halfTanY;

	float rayDirX = cameraForward[0] + su * cameraRight[0] + sv * cameraUp[0];
	float rayDirY = cameraForward[1] + su * cameraRight[1] + sv * cameraUp[1];
	float rayDirZ = cameraForward[2] + su * cameraRight[2] + sv * cameraUp[2];

	float il = rsqrtf(rayDirX * rayDirX + rayDirY * rayDirY + rayDirZ * rayDirZ);
	rayDirX *= il;
	rayDirY *= il;
	rayDirZ *= il;

	float rayOrigX = cameraPos[0];
	float rayOrigY = cameraPos[1];
	float rayOrigZ = cameraPos[2];

	// whitted-style iterative bounce loop
	float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
	float throughR = 1.0f, throughG = 1.0f, throughB = 1.0f;

	const int MAX_BOUNCES = 4;
	const float SPECULAR_POWER = 64.0f;
	const float REFLECTIVITY = 0.15f;
	int alphaSkips = 0;
	const int MAX_ALPHA_SKIPS = 8;
	float biasOffset = fmaxf(rayOffset, 0.01f);

	for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {

		int hitIdx;
		float hitU, hitV;
		float hitT = TraceRay_RT(rayOrigX, rayOrigY, rayOrigZ,
								 rayDirX, rayDirY, rayDirZ,
								 vertices, triangles, triIndices, bvhNodes,
								 hitIdx, hitU, hitV);

		if (hitIdx < 0) {
			// sky / miss — use sky color parameters
			float upDot = rayDirY; // approximate up = Y
			float skyR, skyG, skyB;
			if (upDot > 0.0f) {
				float t = fminf(upDot, 1.0f);
				skyR = skyColorHorizon[0] * (1.0f - t) + skyColorZenith[0] * t;
				skyG = skyColorHorizon[1] * (1.0f - t) + skyColorZenith[1] * t;
				skyB = skyColorHorizon[2] * (1.0f - t) + skyColorZenith[2] * t;
			} else {
				float t = fminf(-upDot, 1.0f);
				skyR = skyColorHorizon[0] * (1.0f - t) + skyColorGround[0] * t;
				skyG = skyColorHorizon[1] * (1.0f - t) + skyColorGround[1] * t;
				skyB = skyColorHorizon[2] * (1.0f - t) + skyColorGround[2] * t;
			}
			colorR += throughR * skyR * skyIntensity;
			colorG += throughG * skyG * skyIntensity;
			colorB += throughB * skyB * skyIntensity;
			break;
		}

		const ptTriangle_t& tri = triangles[hitIdx];
		int i0 = tri.vertexIndices[0];
		int i1 = tri.vertexIndices[1];
		int i2 = tri.vertexIndices[2];

		float w0 = 1.0f - hitU - hitV;
		float w1 = hitU;
		float w2 = hitV;

		// interpolate surface normal
		float nx = w0 * vertices[i0].normal[0] + w1 * vertices[i1].normal[0] + w2 * vertices[i2].normal[0];
		float ny = w0 * vertices[i0].normal[1] + w1 * vertices[i1].normal[1] + w2 * vertices[i2].normal[1];
		float nz = w0 * vertices[i0].normal[2] + w1 * vertices[i1].normal[2] + w2 * vertices[i2].normal[2];
		float nIL = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
		nx *= nIL;
		ny *= nIL;
		nz *= nIL;

		// interpolate texcoord
		float tu = w0 * vertices[i0].texcoord[0] + w1 * vertices[i1].texcoord[0] + w2 * vertices[i2].texcoord[0];
		float tv = w0 * vertices[i0].texcoord[1] + w1 * vertices[i1].texcoord[1] + w2 * vertices[i2].texcoord[1];

		// hit position
		float pX = rayOrigX + rayDirX * hitT;
		float pY = rayOrigY + rayDirY * hitT;
		float pZ = rayOrigZ + rayDirZ * hitT;

		// material lookup
		const ptMaterial_t& mat = materials[tri.materialIndex];
		float albR = mat.albedo[0];
		float albG = mat.albedo[1];
		float albB = mat.albedo[2];
		float alpha = 1.0f;

		if (mat.albedoTexture >= 0) {
			float texColor[4];
			SampleTexture(textures, mat.albedoTexture, tu, tv, texColor);
			albR *= texColor[0];
			albG *= texColor[1];
			albB *= texColor[2];
			alpha = texColor[3];
		}

		// alpha test: perforated surface fails — skip through
		if (mat.alphaTest > 0.0f && alpha < mat.alphaTest && alphaSkips < MAX_ALPHA_SKIPS) {
			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// nearly invisible alpha blend — skip through
		if (mat.blendMode == 1 && alpha < 0.05f && alphaSkips < MAX_ALPHA_SKIPS) {
			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// lit noshadows decal overlays — blend over and continue
		if (mat.noShadows && !mat.isAmbientOnly && mat.alphaTest == 0.0f
			&& mat.transmission == 0.0f && mat.blendMode == 0 && alphaSkips < MAX_ALPHA_SKIPS) {
			float lum = albR * 0.2126f + albG * 0.7152f + albB * 0.0722f;
			float decalAlpha = (lum < 0.02f) ? 0.0f : alpha;

			if (decalAlpha > 0.05f) {
				colorR += throughR * albR * decalAlpha * 0.5f;
				colorG += throughG * albG * decalAlpha * 0.5f;
				colorB += throughB * albB * decalAlpha * 0.5f;
				throughR *= (1.0f - decalAlpha);
				throughG *= (1.0f - decalAlpha);
				throughB *= (1.0f - decalAlpha);
			}

			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// ambient-only materials (self-illuminated overlays, particles, HUD)
		if (mat.isAmbientOnly && alphaSkips < MAX_ALPHA_SKIPS) {
			if (mat.blendMode == 2) {
				// additive
				colorR += throughR * albR * alpha * emissionBoost;
				colorG += throughG * albG * alpha * emissionBoost;
				colorB += throughB * albB * alpha * emissionBoost;
			} else if (mat.blendMode == 3) {
				// modulate
				throughR *= albR;
				throughG *= albG;
				throughB *= albB;
			} else {
				// default alpha-over
				colorR += throughR * albR * alpha;
				colorG += throughG * albG * alpha;
				colorB += throughB * albB * alpha;
				throughR *= (1.0f - alpha);
				throughG *= (1.0f - alpha);
				throughB *= (1.0f - alpha);
			}

			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// additive blending (non-ambient)
		if (mat.blendMode == 2 && alphaSkips < MAX_ALPHA_SKIPS) {
			colorR += throughR * albR * alpha;
			colorG += throughG * albG * alpha;
			colorB += throughB * albB * alpha;

			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// modulate blending (non-ambient)
		if (mat.blendMode == 3 && alphaSkips < MAX_ALPHA_SKIPS) {
			throughR *= albR;
			throughG *= albG;
			throughB *= albB;

			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// alpha blend (non-ambient) — partial transparency
		if (mat.blendMode == 1 && alpha < 0.95f && alphaSkips < MAX_ALPHA_SKIPS) {
			colorR += throughR * albR * alpha;
			colorG += throughG * albG * alpha;
			colorB += throughB * albB * alpha;
			throughR *= (1.0f - alpha);
			throughG *= (1.0f - alpha);
			throughB *= (1.0f - alpha);

			rayOrigX = pX + rayDirX * biasOffset;
			rayOrigY = pY + rayDirY * biasOffset;
			rayOrigZ = pZ + rayDirZ * biasOffset;
			bounce--;
			alphaSkips++;
			continue;
		}

		// emission contribution
		float emitR = mat.emission[0] * emissionBoost;
		float emitG = mat.emission[1] * emissionBoost;
		float emitB = mat.emission[2] * emissionBoost;
		float emitLum = emitR * 0.2126f + emitG * 0.7152f + emitB * 0.0722f;
		if (emitLum > 0.001f) {
			colorR += throughR * emitR;
			colorG += throughG * emitG;
			colorB += throughB * emitB;
			if (emitLum > 0.5f) break;
		}

		// view direction (toward camera)
		float vX = -rayDirX;
		float vY = -rayDirY;
		float vZ = -rayDirZ;

		// direct lighting from all lights
		float diffR = 0.0f, diffG = 0.0f, diffB = 0.0f;
		float specR = 0.0f, specG = 0.0f, specB = 0.0f;

		for (int li = 0; li < numLights; li++) {
			const ptLight_t& light = lights[li];

			float lX, lY, lZ;
			float shadowDist;
			float atten = 0.0f;
			float lightColR = light.color[0];
			float lightColG = light.color[1];
			float lightColB = light.color[2];

			if (light.type == 0) {
				// point light
				float toL_x = light.position[0] - pX;
				float toL_y = light.position[1] - pY;
				float toL_z = light.position[2] - pZ;

				float dSq = toL_x * toL_x + toL_y * toL_y + toL_z * toL_z;
				float d = sqrtf(dSq + 1e-12f);
				float iD = 1.0f / d;
				lX = toL_x * iD;
				lY = toL_y * iD;
				lZ = toL_z * iD;
				shadowDist = d;

				float R = light.radius;
				if (d >= R) continue;

				float ratio = d / R;
				float falloff = 1.0f - ratio * ratio;
				falloff = falloff * falloff;
				atten = light.intensity * falloff;

			} else if (light.type == 1) {
				// directional light
				lX = light.direction[0];
				lY = light.direction[1];
				lZ = light.direction[2];

				float dIL = rsqrtf(lX * lX + lY * lY + lZ * lZ + 1e-12f);
				lX *= dIL;
				lY *= dIL;
				lZ *= dIL;

				shadowDist = 10000.0f;
				atten = light.intensity;

			} else {
				// projected / spot light (type == 2)
				float toL_x = light.position[0] - pX;
				float toL_y = light.position[1] - pY;
				float toL_z = light.position[2] - pZ;

				float dSq = toL_x * toL_x + toL_y * toL_y + toL_z * toL_z;
				float d = sqrtf(dSq + 1e-12f);
				float iD = 1.0f / d;
				lX = toL_x * iD;
				lY = toL_y * iD;
				lZ = toL_z * iD;
				shadowDist = d;

				float R = light.radius;
				if (d >= R) continue;

				float ratio = d / R;
				float falloff = 1.0f - ratio * ratio;
				falloff = falloff * falloff;
				float distAtten = falloff;

				// cone attenuation
				float negLX = -lX, negLY = -lY, negLZ = -lZ;
				float spotCos = negLX * light.direction[0] + negLY * light.direction[1] + negLZ * light.direction[2];
				float cosHalfAngle = cosf(light.coneAngle);

				float coneAtten = 0.0f;
				if (spotCos > cosHalfAngle) {
					float t = (spotCos - cosHalfAngle) / (1.0f - cosHalfAngle + 1e-6f);
					coneAtten = powf(t, light.coneFalloff);
				} else {
					continue;
				}

				atten = light.intensity * distAtten * coneAtten;

				// projected texture modulation
				if (light.projectedTextureIndex >= 0) {
					float localX = toL_x * light.right[0] + toL_y * light.right[1] + toL_z * light.right[2];
					float localY = toL_x * light.up[0]    + toL_y * light.up[1]    + toL_z * light.up[2];
					float localZ = toL_x * light.direction[0] + toL_y * light.direction[1] + toL_z * light.direction[2];

					if (localZ > 0.001f) {
						float projU = fmaxf(0.0f, fminf(1.0f, (localX / localZ) * 0.5f + 0.5f));
						float projV = fmaxf(0.0f, fminf(1.0f, (localY / localZ) * 0.5f + 0.5f));

						float projColor[4];
						SampleTexture(textures, light.projectedTextureIndex, projU, projV, projColor);
						lightColR *= projColor[0];
						lightColG *= projColor[1];
						lightColB *= projColor[2];
					} else {
						continue;
					}
				}
			}

			if (atten <= 0.0f) continue;

			// shadow ray
			float sOx = pX + nx * biasOffset;
			float sOy = pY + ny * biasOffset;
			float sOz = pZ + nz * biasOffset;

			if (TraceShadowRay_RT(sOx, sOy, sOz, lX, lY, lZ, shadowDist,
								  vertices, triangles, triIndices, bvhNodes)) {
				continue;
			}

			// Lambertian diffuse
			float NdotL = fmaxf(nx * lX + ny * lY + nz * lZ, 0.0f);

			diffR += lightColR * NdotL * atten;
			diffG += lightColG * NdotL * atten;
			diffB += lightColB * NdotL * atten;

			// Blinn-Phong specular
			float hX = lX + vX;
			float hY = lY + vY;
			float hZ = lZ + vZ;
			float hIL = rsqrtf(hX * hX + hY * hY + hZ * hZ + 1e-12f);
			hX *= hIL;
			hY *= hIL;
			hZ *= hIL;

			float NdotH = fmaxf(nx * hX + ny * hY + nz * hZ, 0.0f);
			float spec = powf(NdotH, SPECULAR_POWER);

			specR += lightColR * spec * atten;
			specG += lightColG * spec * atten;
			specB += lightColB * spec * atten;
		}

		// combine: diffuse * albedo + specular (no ambient — Doom 3 pure darkness)
		float localR = albR * diffR + REFLECTIVITY * specR * specularBoost;
		float localG = albG * diffG + REFLECTIVITY * specG * specularBoost;
		float localB = albB * diffB + REFLECTIVITY * specB * specularBoost;

		// add local contribution weighted by throughput
		float oneMinusRefl = 1.0f - REFLECTIVITY;
		colorR += throughR * localR * oneMinusRefl;
		colorG += throughG * localG * oneMinusRefl;
		colorB += throughB * localB * oneMinusRefl;

		// Reflection ray for next bounce
		float RdotN = rayDirX * nx + rayDirY * ny + rayDirZ * nz;
		rayDirX = rayDirX - 2.0f * RdotN * nx;
		rayDirY = rayDirY - 2.0f * RdotN * ny;
		rayDirZ = rayDirZ - 2.0f * RdotN * nz;

		rayOrigX = pX + nx * biasOffset;
		rayOrigY = pY + ny * biasOffset;
		rayOrigZ = pZ + nz * biasOffset;

		// attenuate throughput by reflectivity
		throughR *= REFLECTIVITY * albR;
		throughG *= REFLECTIVITY * albG;
		throughB *= REFLECTIVITY * albB;

		// early out if throughput is negligible
		if (throughR + throughG + throughB < earlyTermThreshold) {
			break;
		}
	}

	// write HDR framebuffer (tone mapping kernel handles LDR conversion)
	framebuffer[pixelIndex + 0] = colorR;
	framebuffer[pixelIndex + 1] = colorG;
	framebuffer[pixelIndex + 2] = colorB;
	framebuffer[pixelIndex + 3] = 1.0f;
}
