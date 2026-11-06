#ifndef __CU_RENDERER_SAMPLING_CUH__
#define __CU_RENDERER_SAMPLING_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"
#include "renderer_pt/cuda/pt_cuda_material.cuh"

/*
=================================================================================
Device variables for configurable parameters (set by host before kernel launch)
=================================================================================
*/
__device__ float g_softShadowScale = 0.0f;
__device__ float g_rayOffset = 0.02f;

/*
=================================================================================
Cosine-weighted hemisphere sampling
=================================================================================
*/
__device__ void SampleHemisphere(
	const float* normal,
	curandState* randState,
	float* direction,
	float& pdf
) {
	float r1 = curand_uniform(randState);
	float r2 = curand_uniform(randState);

	float sinTheta = sqrtf(r1);
	float cosTheta = sqrtf(1.0f - r1);
	float phi = 2.0f * PT_PI * r2;

	// build orthonormal basis from normal
	float upX, upY, upZ;
	if (fabsf(normal[0]) < 0.9f) {
		upX = 1.0f; upY = 0.0f; upZ = 0.0f;
	} else {
		upX = 0.0f; upY = 1.0f; upZ = 0.0f;
	}

	// tangent = normalize(cross(up, normal))
	float tX = upY * normal[2] - upZ * normal[1];
	float tY = upZ * normal[0] - upX * normal[2];
	float tZ = upX * normal[1] - upY * normal[0];
	float tIL = rsqrtf(tX * tX + tY * tY + tZ * tZ + 1e-12f);
	tX *= tIL; tY *= tIL; tZ *= tIL;

	// bitangent = cross(normal, tangent)
	float bX = normal[1] * tZ - normal[2] * tY;
	float bY = normal[2] * tX - normal[0] * tZ;
	float bZ = normal[0] * tY - normal[1] * tX;

	float cosP = cosf(phi);
	float sinP = sinf(phi);

	direction[0] = sinTheta * cosP * tX + sinTheta * sinP * bX + cosTheta * normal[0];
	direction[1] = sinTheta * cosP * tY + sinTheta * sinP * bY + cosTheta * normal[1];
	direction[2] = sinTheta * cosP * tZ + sinTheta * sinP * bZ + cosTheta * normal[2];

	// PDF for cosine-weighted sampling: cosTheta / pi
	pdf = cosTheta / PT_PI;
}

/*
=================================================================================
Henyey-Greenstein phase function for volumetric scattering
=================================================================================
*/
__device__ inline float HenyeyGreenstein(float cosTheta, float g) {
	float g2 = g * g;
	float denom = 1.0f + g2 - 2.0f * g * cosTheta;
	denom = fmaxf(denom, 1e-6f);
	return (1.0f - g2) / (4.0f * PT_PI * powf(denom, 1.5f));
}

/*
=================================================================================
Spotlight attenuation (smoothstep cone)
=================================================================================
*/
__device__ float SpotlightAttenuation(const ptLight_t& light, const float* toSurface) {
	// compute angle between the vector from light to surface and the light direction
	float cosAngle = dot3(toSurface, light.direction);

	float cosCone = cosf(light.coneAngle);
	float cosInner = cosf(light.coneAngle * 0.8f);  // Inner cone = 80% of outer

	if (cosAngle < cosCone) {
		return 0.0f;  // Outside cone
	}
	if (cosAngle > cosInner) {
		return 1.0f;  // Inside inner cone (full brightness)
	}

	// smoothstep between inner and outer cone
	float t = (cosAngle - cosCone) / (cosInner - cosCone);
	t = fmaxf(0.0f, fminf(1.0f, t));
	// smooth hermite interpolation
	float atten = t * t * (3.0f - 2.0f * t);

	// apply falloff exponent for sharper/softer edges
	if (light.coneFalloff > 0.0f) {
		atten = powf(atten, light.coneFalloff);
	}

	return atten;
}

/*
=================================================================================
SampleLightProjectionUV

Compute projected texture UV coordinates using Doom 3's lightProject planes.
=================================================================================
*/
__device__ bool SampleLightProjectionUV(
	const ptLight_t& light,
	const float* worldPos,
	float& u,
	float& v,
	float& falloff
) {
	float s = light.lightProject[0][0] * worldPos[0] +
			  light.lightProject[0][1] * worldPos[1] +
			  light.lightProject[0][2] * worldPos[2] +
			  light.lightProject[0][3];

	float t = light.lightProject[1][0] * worldPos[0] +
			  light.lightProject[1][1] * worldPos[1] +
			  light.lightProject[1][2] * worldPos[2] +
			  light.lightProject[1][3];

	float q = light.lightProject[2][0] * worldPos[0] +
			  light.lightProject[2][1] * worldPos[1] +
			  light.lightProject[2][2] * worldPos[2] +
			  light.lightProject[2][3];

	float f = light.lightProject[3][0] * worldPos[0] +
			  light.lightProject[3][1] * worldPos[1] +
			  light.lightProject[3][2] * worldPos[2] +
			  light.lightProject[3][3];

	if (q <= 0.0001f) {
		return false;
	}

	float invQ = 1.0f / q;
	u = s * invQ;
	v = t * invQ;
	falloff = f;

	if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
		return false;
	}
	if (falloff < 0.0f || falloff > 1.0f) {
		return false;
	}

	return true;
}

/*
=================================================================================
JitterLightPosition - Area light sampling for soft shadows
=================================================================================
*/
__device__ void JitterLightPosition(
	const ptLight_t& light,
	curandState* randState,
	float* jitteredPos
) {
	jitteredPos[0] = light.position[0];
	jitteredPos[1] = light.position[1];
	jitteredPos[2] = light.position[2];

	float jitterScale = g_softShadowScale;
	if (jitterScale <= 0.0f) {
		return;
	}

	if (light.type == 2) {
		return;  // area light already a point
	}
	if (light.type == 1) {
		return;  // directional light
	}

	if (light.coneAngle > 0.0f) {
		float r1 = (curand_uniform(randState) - 0.5f) * 2.0f * jitterScale;
		float r2 = (curand_uniform(randState) - 0.5f) * 2.0f * jitterScale;

		float rightLen = light.right[0] * light.right[0] + light.right[1] * light.right[1] + light.right[2] * light.right[2];
		if (rightLen > 0.001f) {
			float extent = light.radius * jitterScale;
			jitteredPos[0] += light.right[0] * r1 * extent + light.up[0] * r2 * extent;
			jitteredPos[1] += light.right[1] * r1 * extent + light.up[1] * r2 * extent;
			jitteredPos[2] += light.right[2] * r1 * extent + light.up[2] * r2 * extent;
		}
	} else {
		float r1 = (curand_uniform(randState) - 0.5f) * 2.0f;
		float r2 = (curand_uniform(randState) - 0.5f) * 2.0f;
		float r3 = (curand_uniform(randState) - 0.5f) * 2.0f;

		jitteredPos[0] += r1 * light.lightRadius3[0] * jitterScale;
		jitteredPos[1] += r2 * light.lightRadius3[1] * jitterScale;
		jitteredPos[2] += r3 * light.lightRadius3[2] * jitterScale;
	}
}

/*
=================================================================================
SampleVolumetricScattering
=================================================================================
*/
__device__ void SampleVolumetricScattering(
	const ptLight_t& light,
	const float* rayOrigin,
	const float* rayDirection,
	float rayLength,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* nodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	curandState* randState,
	float* scattering,
	float volDensity,
	int volSteps,
	float volAnisotropy,
	float volFalloff,
	float volMaxDist
) {
	scattering[0] = scattering[1] = scattering[2] = 0.0f;

	if (light.volumetric <= 0.0f) {
		return;
	}

	int steps = max(2, min(64, volSteps));
	float density = fmaxf(0.0001f, volDensity) * light.volumetric;
	float g = fmaxf(-0.99f, fminf(0.99f, volAnisotropy));

	if (light.coneAngle > 0.0f) {
		g = fmaxf(g, 0.7f);
	}

	float marchLength = rayLength;
	if (volMaxDist > 0.0f && marchLength > volMaxDist) {
		marchLength = volMaxDist;
	}

	float stepSize = marchLength / (float)steps;
	float transmittance = 1.0f;
	float extinctionCoeff = density * 0.5f;

	for (int step = 0; step < steps; step++) {
		float t = (step + curand_uniform(randState)) * stepSize;
		float samplePos[3] = {
			rayOrigin[0] + rayDirection[0] * t,
			rayOrigin[1] + rayDirection[1] * t,
			rayOrigin[2] + rayDirection[2] * t
		};

		float stepTransmit = expf(-extinctionCoeff * stepSize);

		if (transmittance < 0.01f) {
			break;
		}

		float lightDir[3];
		float lightDist;

		if (light.type == 1) {
			lightDir[0] = -light.direction[0];
			lightDir[1] = -light.direction[1];
			lightDir[2] = -light.direction[2];
			lightDist = 1e10f;
		} else {
			lightDir[0] = light.position[0] - samplePos[0];
			lightDir[1] = light.position[1] - samplePos[1];
			lightDir[2] = light.position[2] - samplePos[2];
			lightDist = length3(lightDir);

			if (lightDist < 1e-4f) { transmittance *= stepTransmit; continue; }

			lightDir[0] /= lightDist;
			lightDir[1] /= lightDist;
			lightDir[2] /= lightDist;
		}

		float spotAtten = 1.0f;
		if (light.projectedTextureIndex < 0 && light.coneAngle > 0.0f) {
			float toSample[3] = {-lightDir[0], -lightDir[1], -lightDir[2]};
			spotAtten = SpotlightAttenuation(light, toSample);
			if (spotAtten < 0.001f) { transmittance *= stepTransmit; continue; }
		}

		// shadow ray from sample to light
		ptRay_t shadowRay;
		shadowRay.origin[0] = samplePos[0];
		shadowRay.origin[1] = samplePos[1];
		shadowRay.origin[2] = samplePos[2];
		shadowRay.direction[0] = lightDir[0];
		shadowRay.direction[1] = lightDir[1];
		shadowRay.direction[2] = lightDir[2];
		shadowRay.tMin = g_rayOffset;
		shadowRay.tMax = lightDist - g_rayOffset;

		float shadowAtten = 1.0f;
		const int MAX_VOL_SHADOW_STEPS = 4;

		for (int ss = 0; ss < MAX_VOL_SHADOW_STEPS; ss++) {
			ptHitInfo_t shadowHit;
			shadowHit.hit = false;
			shadowHit.t = shadowRay.tMax;

			if (!IntersectBVH(shadowRay, vertices, triangles, nodes, triIndices, materials, shadowHit)) {
				break;
			}

			const ptMaterial_t& hitMat = materials[shadowHit.materialIndex];

			if (hitMat.noShadows || hitMat.isAmbientOnly || hitMat.blendMode == 2) {
				shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
				shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
				shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
				shadowRay.tMax -= shadowHit.t;
				if (shadowRay.tMax < 0.01f) break;
				continue;
			}

			if (hitMat.alphaTest > 0.0f && hitMat.albedoTexture >= 0 && textures) {
				float texAlpha[4];
				SampleTexture(textures, hitMat.albedoTexture, shadowHit.texcoord[0], shadowHit.texcoord[1], texAlpha);
				if (texAlpha[3] * hitMat.albedo[3] < hitMat.alphaTest) {
					shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
					shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
					shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
					shadowRay.tMax -= shadowHit.t;
					if (shadowRay.tMax < 0.01f) break;
					continue;
				}
			}

			if (hitMat.transmission > 0.0f) {
				shadowAtten *= hitMat.transmission * 0.8f;
				if (shadowAtten < 0.01f) { shadowAtten = 0.0f; break; }
				shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
				shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
				shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
				shadowRay.tMax -= shadowHit.t;
				if (shadowRay.tMax < 0.01f) break;
				continue;
			}

			if (hitMat.blendMode == 1) {
				float alpha = hitMat.albedo[3];
				if (hitMat.albedoTexture >= 0 && textures) {
					float tex[4];
					SampleTexture(textures, hitMat.albedoTexture, shadowHit.texcoord[0], shadowHit.texcoord[1], tex);
					alpha *= tex[3];
				}
				shadowAtten *= (1.0f - alpha);
				if (shadowAtten < 0.01f) { shadowAtten = 0.0f; break; }
				shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
				shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
				shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
				shadowRay.tMax -= shadowHit.t;
				if (shadowRay.tMax < 0.01f) break;
				continue;
			}

			shadowAtten = 0.0f;
			break;
		}

		if (shadowAtten < 0.001f) {
			transmittance *= stepTransmit;
			continue;
		}

		// distance attenuation
		float attenuation = 1.0f;
		if (light.type != 1) {
			float lightRadius = fmaxf(light.radius, 1.0f);
			float lt = lightDist / lightRadius;
			if (lt >= 1.0f) {
				transmittance *= stepTransmit;
				continue;
			}
			float window = 1.0f - lt * lt;
			window = window * window;
			attenuation = window;
		}

		// distance falloff
		float distFalloff = 1.0f;
		if (volFalloff > 0.0f && marchLength > 1.0f) {
			float normalizedDist = t / marchLength;
			distFalloff = powf(1.0f - normalizedDist, volFalloff);
		}

		// phase function
		float cosTheta = -(rayDirection[0] * lightDir[0] +
						   rayDirection[1] * lightDir[1] +
						   rayDirection[2] * lightDir[2]);
		float phase = HenyeyGreenstein(cosTheta, g);

		float contribution = light.intensity * density * stepSize * attenuation *
							 phase * shadowAtten * spotAtten * transmittance * distFalloff;
		if (contribution > 1.0f) contribution = 1.0f;

		float volColor[3] = {light.color[0], light.color[1], light.color[2]};
		if (light.projectedTextureIndex >= 0 && textures) {
			float projU, projV, projFalloff;
			if (SampleLightProjectionUV(light, samplePos, projU, projV, projFalloff)) {
				float texSample[4];
				SampleTexture(textures, light.projectedTextureIndex, projU, projV, texSample);
				volColor[0] *= texSample[0];
				volColor[1] *= texSample[1];
				volColor[2] *= texSample[2];
				float projFalloffAtten = 1.0f - projFalloff;
				projFalloffAtten = fmaxf(0.0f, projFalloffAtten * projFalloffAtten);
				contribution *= projFalloffAtten;
			} else {
				transmittance *= stepTransmit;
				continue;
			}
		}

		scattering[0] += volColor[0] * contribution;
		scattering[1] += volColor[1] * contribution;
		scattering[2] += volColor[2] * contribution;

		transmittance *= stepTransmit;
	}
}

/*
=================================================================================
SampleDirectLight - Transparency-aware direct light sampling
=================================================================================
*/
__device__ void SampleDirectLight(
	const ptLight_t& light,
	const float* position,
	const float* normal,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* nodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	curandState* randState,
	float* radiance
) {
	radiance[0] = radiance[1] = radiance[2] = 0.0f;

	// jitter light position for soft shadows
	float lightPos[3];
	JitterLightPosition(light, randState, lightPos);

	float lightDir[3];
	float distance;

	if (light.type == 0 || light.type == 2) {
		lightDir[0] = lightPos[0] - position[0];
		lightDir[1] = lightPos[1] - position[1];
		lightDir[2] = lightPos[2] - position[2];
		distance = length3(lightDir);

		if (distance < 1e-4f) return;

		lightDir[0] /= distance;
		lightDir[1] /= distance;
		lightDir[2] /= distance;
	} else {
		lightDir[0] = -light.direction[0];
		lightDir[1] = -light.direction[1];
		lightDir[2] = -light.direction[2];
		distance = 1e10f;
	}

	float ndotl = dot3(normal, lightDir);
	if (ndotl <= 0.0f) return;

	// projected light frustum clipping
	float spotAtten = 1.0f;
	float cachedProjU = 0.0f, cachedProjV = 0.0f, cachedProjFalloff = 0.0f;
	bool hasProjectionUV = false;

	if (light.projectedTextureIndex >= 0) {
		if (!SampleLightProjectionUV(light, position, cachedProjU, cachedProjV, cachedProjFalloff)) {
			return;
		}
		hasProjectionUV = true;
	} else if (light.coneAngle > 0.0f) {
		float toSurface[3];
		if (light.type == 0 || light.type == 2) {
			toSurface[0] = -lightDir[0];
			toSurface[1] = -lightDir[1];
			toSurface[2] = -lightDir[2];
		} else {
			toSurface[0] = lightDir[0];
			toSurface[1] = lightDir[1];
			toSurface[2] = lightDir[2];
		}
		spotAtten = SpotlightAttenuation(light, toSurface);
		if (spotAtten < 0.001f) return;
	}

	// shadow ray
	ptRay_t shadowRay;
	shadowRay.origin[0] = position[0] + normal[0] * g_rayOffset;
	shadowRay.origin[1] = position[1] + normal[1] * g_rayOffset;
	shadowRay.origin[2] = position[2] + normal[2] * g_rayOffset;
	shadowRay.direction[0] = lightDir[0];
	shadowRay.direction[1] = lightDir[1];
	shadowRay.direction[2] = lightDir[2];
	shadowRay.tMin = g_rayOffset;
	shadowRay.tMax = distance - g_rayOffset;

	// transparency-aware shadow
	float shadowAtten[3] = {1.0f, 1.0f, 1.0f};
	const int MAX_SHADOW_STEPS = 8;

	for (int step = 0; step < MAX_SHADOW_STEPS; step++) {
		ptHitInfo_t shadowHit;
		shadowHit.hit = false;
		shadowHit.t = shadowRay.tMax;

		if (!IntersectBVH(shadowRay, vertices, triangles, nodes, triIndices, materials, shadowHit)) {
			break;
		}

		const ptMaterial_t& hitMat = materials[shadowHit.materialIndex];

		if (hitMat.noShadows) {
			shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
			shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
			shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
			shadowRay.tMax -= shadowHit.t;
			if (shadowRay.tMax < 0.01f) break;
			continue;
		}

		if (hitMat.blendMode == 2) {
			shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
			shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
			shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
			shadowRay.tMax -= shadowHit.t;
			if (shadowRay.tMax < 0.01f) break;
			continue;
		}

		if (hitMat.isAmbientOnly) {
			shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
			shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
			shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
			shadowRay.tMax -= shadowHit.t;
			if (shadowRay.tMax < 0.01f) break;
			continue;
		}

		if (hitMat.alphaTest > 0.0f && hitMat.albedoTexture >= 0 && textures) {
			float texAlpha[4];
			SampleTexture(textures, hitMat.albedoTexture, shadowHit.texcoord[0], shadowHit.texcoord[1], texAlpha);
			float alpha = texAlpha[3] * hitMat.albedo[3];
			if (alpha < hitMat.alphaTest) {
				shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
				shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
				shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
				shadowRay.tMax -= shadowHit.t;
				if (shadowRay.tMax < 0.01f) break;
				continue;
			}
		}

		if (hitMat.transmission > 0.0f) {
			float negDir[3] = {-shadowRay.direction[0], -shadowRay.direction[1], -shadowRay.direction[2]};
			float cosT = fabsf(dot3(shadowHit.normal, negDir));
			float r0 = (hitMat.ior - 1.0f) / (hitMat.ior + 1.0f);
			r0 = r0 * r0;
			float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosT, 5.0f);
			float transmitFrac = (1.0f - fresnel) * hitMat.transmission;

			float tint[4];
			if (hitMat.albedoTexture >= 0 && textures) {
				SampleTexture(textures, hitMat.albedoTexture, shadowHit.texcoord[0], shadowHit.texcoord[1], tint);
				tint[0] *= hitMat.albedo[0];
				tint[1] *= hitMat.albedo[1];
				tint[2] *= hitMat.albedo[2];
			} else {
				tint[0] = hitMat.albedo[0];
				tint[1] = hitMat.albedo[1];
				tint[2] = hitMat.albedo[2];
			}

			shadowAtten[0] *= transmitFrac * tint[0];
			shadowAtten[1] *= transmitFrac * tint[1];
			shadowAtten[2] *= transmitFrac * tint[2];

			if (shadowAtten[0] < 0.01f && shadowAtten[1] < 0.01f && shadowAtten[2] < 0.01f) {
				return;
			}

			shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
			shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
			shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
			shadowRay.tMax -= shadowHit.t;
			if (shadowRay.tMax < 0.01f) break;
			continue;
		}

		if (hitMat.blendMode == 1) {
			float alpha = hitMat.albedo[3];
			if (hitMat.albedoTexture >= 0 && textures) {
				float texAlpha[4];
				SampleTexture(textures, hitMat.albedoTexture, shadowHit.texcoord[0], shadowHit.texcoord[1], texAlpha);
				alpha *= texAlpha[3];
			}
			shadowAtten[0] *= (1.0f - alpha);
			shadowAtten[1] *= (1.0f - alpha);
			shadowAtten[2] *= (1.0f - alpha);

			if (shadowAtten[0] < 0.01f && shadowAtten[1] < 0.01f && shadowAtten[2] < 0.01f) {
				return;
			}

			shadowRay.origin[0] = shadowHit.position[0] + shadowRay.direction[0] * 0.002f;
			shadowRay.origin[1] = shadowHit.position[1] + shadowRay.direction[1] * 0.002f;
			shadowRay.origin[2] = shadowHit.position[2] + shadowRay.direction[2] * 0.002f;
			shadowRay.tMax -= shadowHit.t;
			if (shadowRay.tMax < 0.01f) break;
			continue;
		}

		// opaque
		return;
	}

	// distance attenuation (Doom 3 windowed falloff)
	float attenuation = 1.0f;
	if (light.type == 0 || light.type == 2) {
		float lightRadius = fmaxf(light.radius, 1.0f);
		float t = distance / lightRadius;
		if (t >= 1.0f) return;
		float window = 1.0f - t * t;
		window = window * window;
		attenuation = window;
	}

	float contribution = light.intensity * ndotl * attenuation;
	if (contribution > 10.0f) contribution = 10.0f;

	// light texture sampling
	float lightColor[3] = {light.color[0], light.color[1], light.color[2]};

	if (hasProjectionUV && textures) {
		float texSample[4];
		SampleTexture(textures, light.projectedTextureIndex, cachedProjU, cachedProjV, texSample);
		lightColor[0] *= texSample[0];
		lightColor[1] *= texSample[1];
		lightColor[2] *= texSample[2];

		float falloffAtten = 1.0f - cachedProjFalloff;
		falloffAtten = fmaxf(0.0f, falloffAtten * falloffAtten);
		contribution *= falloffAtten;
	} else if (light.textureIndex >= 0 && textures) {
		float texSample[4];
		SampleTexture(textures, light.textureIndex, 0.5f, 0.5f, texSample);
		texSample[0] = fminf(texSample[0], 1.0f);
		texSample[1] = fminf(texSample[1], 1.0f);
		texSample[2] = fminf(texSample[2], 1.0f);
		lightColor[0] *= texSample[0];
		lightColor[1] *= texSample[1];
		lightColor[2] *= texSample[2];
	}

	radiance[0] = lightColor[0] * contribution * shadowAtten[0] * spotAtten;
	radiance[1] = lightColor[1] * contribution * shadowAtten[1] * spotAtten;
	radiance[2] = lightColor[2] * contribution * shadowAtten[2] * spotAtten;
}

/*
=================================================================================
SampleIndirectLight - One-bounce global illumination
=================================================================================
*/
__device__ void SampleIndirectLight(
	const float* position,
	const float* normal,
	const float* direction,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* bvhNodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	const ptLight_t* lights,
	int numLights,
	curandState* randState,
	float* radiance
) {
	radiance[0] = radiance[1] = radiance[2] = 0.0f;

	ptRay_t ray;
	ray.origin[0] = position[0] + normal[0] * g_rayOffset;
	ray.origin[1] = position[1] + normal[1] * g_rayOffset;
	ray.origin[2] = position[2] + normal[2] * g_rayOffset;
	ray.direction[0] = direction[0];
	ray.direction[1] = direction[1];
	ray.direction[2] = direction[2];
	ray.tMin = g_rayOffset;
	ray.tMax = 1e10f;

	ptHitInfo_t hitInfo;
	hitInfo.hit = false;
	hitInfo.t = ray.tMax;

	const int MAX_INDIRECT_STEPS = 4;
	bool foundOpaque = false;

	for (int step = 0; step < MAX_INDIRECT_STEPS; step++) {
		hitInfo.hit = false;
		hitInfo.t = ray.tMax;

		if (!IntersectBVH(ray, vertices, triangles, bvhNodes, triIndices, materials, hitInfo)) {
			break;
		}

		const ptMaterial_t& mat = materials[hitInfo.materialIndex];

		if (mat.noShadows) {
			ray.origin[0] = hitInfo.position[0] + ray.direction[0] * 0.002f;
			ray.origin[1] = hitInfo.position[1] + ray.direction[1] * 0.002f;
			ray.origin[2] = hitInfo.position[2] + ray.direction[2] * 0.002f;
			ray.tMin = g_rayOffset;
			continue;
		}

		if (mat.isAmbientOnly) {
			ray.origin[0] = hitInfo.position[0] + ray.direction[0] * 0.002f;
			ray.origin[1] = hitInfo.position[1] + ray.direction[1] * 0.002f;
			ray.origin[2] = hitInfo.position[2] + ray.direction[2] * 0.002f;
			ray.tMin = g_rayOffset;
			continue;
		}

		if (mat.alphaTest > 0.0f && mat.albedoTexture >= 0 && textures) {
			float texSample[4];
			SampleTexture(textures, mat.albedoTexture, hitInfo.texcoord[0], hitInfo.texcoord[1], texSample);
			float alpha = texSample[3] * mat.albedo[3];
			if (alpha < mat.alphaTest) {
				ray.origin[0] = hitInfo.position[0] + ray.direction[0] * 0.002f;
				ray.origin[1] = hitInfo.position[1] + ray.direction[1] * 0.002f;
				ray.origin[2] = hitInfo.position[2] + ray.direction[2] * 0.002f;
				ray.tMin = g_rayOffset;
				continue;
			}
		}

		if (mat.transmission > 0.5f) {
			ray.origin[0] = hitInfo.position[0] + ray.direction[0] * 0.002f;
			ray.origin[1] = hitInfo.position[1] + ray.direction[1] * 0.002f;
			ray.origin[2] = hitInfo.position[2] + ray.direction[2] * 0.002f;
			ray.tMin = g_rayOffset;
			continue;
		}

		foundOpaque = true;
		break;
	}

	if (!foundOpaque || !hitInfo.hit) {
		return;
	}

	ptMaterial_t material = materials[hitInfo.materialIndex];

	float albedo[4] = {material.albedo[0], material.albedo[1], material.albedo[2], material.albedo[3]};
	if (material.albedoTexture >= 0) {
		float texSample[4];
		SampleTexture(textures, material.albedoTexture, hitInfo.texcoord[0], hitInfo.texcoord[1], texSample);
		albedo[0] *= texSample[0];
		albedo[1] *= texSample[1];
		albedo[2] *= texSample[2];
		albedo[3] *= texSample[3];
	}

	// add emission
	radiance[0] += material.emission[0];
	radiance[1] += material.emission[1];
	radiance[2] += material.emission[2];

	// one-bounce direct lighting
	if (numLights > 0) {
		int lightIndex = (int)(curand_uniform(randState) * numLights);
		if (lightIndex >= numLights) lightIndex = numLights - 1;

		float directLight[3];
		SampleDirectLight(lights[lightIndex], hitInfo.position, hitInfo.normal,
			vertices, triangles, bvhNodes, triIndices, materials, textures, randState, directLight);

		float lightWeight = (float)numLights;
		radiance[0] += albedo[0] * directLight[0] * lightWeight / PT_PI;
		radiance[1] += albedo[1] * directLight[1] * lightWeight / PT_PI;
		radiance[2] += albedo[2] * directLight[2] * lightWeight / PT_PI;
	}

	// clamp indirect
	float maxIndirect = 1.5f;
	radiance[0] = fminf(radiance[0], maxIndirect);
	radiance[1] = fminf(radiance[1], maxIndirect);
	radiance[2] = fminf(radiance[2], maxIndirect);
}

#endif // __CU_RENDERER_SAMPLING_CUH__
