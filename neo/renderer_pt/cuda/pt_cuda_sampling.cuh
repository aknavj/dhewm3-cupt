#ifndef __CU_RENDERER_SAMPLING_CUH__
#define __CU_RENDERER_SAMPLING_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
	ptRng_t* rng,
	float* direction,
	float& pdf
) {
	float r1 = PT_Rand(rng);
	float r2 = PT_Rand(rng);

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
	ptRng_t* rng,
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
		float r1 = (PT_Rand(rng) - 0.5f) * 2.0f * jitterScale;
		float r2 = (PT_Rand(rng) - 0.5f) * 2.0f * jitterScale;

		float rightLen = light.right[0] * light.right[0] + light.right[1] * light.right[1] + light.right[2] * light.right[2];
		if (rightLen > 0.001f) {
			float extent = light.radius * jitterScale;
			jitteredPos[0] += light.right[0] * r1 * extent + light.up[0] * r2 * extent;
			jitteredPos[1] += light.right[1] * r1 * extent + light.up[1] * r2 * extent;
			jitteredPos[2] += light.right[2] * r1 * extent + light.up[2] * r2 * extent;
		}
	} else {
		float r1 = (PT_Rand(rng) - 0.5f) * 2.0f;
		float r2 = (PT_Rand(rng) - 0.5f) * 2.0f;
		float r3 = (PT_Rand(rng) - 0.5f) * 2.0f;

		jitteredPos[0] += r1 * light.lightRadius3[0] * jitterScale;
		jitteredPos[1] += r2 * light.lightRadius3[1] * jitterScale;
		jitteredPos[2] += r3 * light.lightRadius3[2] * jitterScale;
	}
}

/*
=================================================================================
PT_LightFilter

Everything about a light's reach that does not depend on visibility: the
light color filtered by the light image, times the returned scalar
attenuation. Returns 0 when the point is outside the light.

boxFalloff lights reproduce the GL interaction: lightProject gives (s, t)
for the light image and the falloff coordinate for the falloff image, both
zero outside the light volume. The other lights use a spherical window
(area lights) and optionally a spot cone.
=================================================================================
*/
__device__ float PT_LightFilter(
	const ptLight_t& light,
	const float* pos,
	float dist,
	const float* toLight,
	const cudaTexture_t* textures,
	float* filter
) {
	filter[0] = light.color[0];
	filter[1] = light.color[1];
	filter[2] = light.color[2];

	if (light.type == PT_LIGHT_DIRECTIONAL) {
		return 1.0f;
	}

	if (light.boxFalloff) {
		float s, t, f;
		if (!SampleLightProjectionUV(light, pos, s, t, f)) {
			return 0.0f;
		}
		float tex[4];
		if (light.projectedTextureIndex >= 0 && textures) {
			SampleTextureSrgb(textures, light.projectedTextureIndex, s, t, tex);
			filter[0] *= tex[0];
			filter[1] *= tex[1];
			filter[2] *= tex[2];
		}
		if (light.falloffTextureIndex >= 0 && textures) {
			// clamp instead of wrapping into the other end of the falloff ramp
			const cudaTexture_t& ft = textures[light.falloffTextureIndex];
			const float halfTexel = ft.width > 0 ? 0.5f / (float)ft.width : 0.0f;
			SampleTexture(textures, light.falloffTextureIndex, fminf(fmaxf(f, halfTexel), 1.0f - halfTexel), 0.5f, tex);
			return PT_SrgbToLinear(tex[0]);
		}
		return 1.0f;
	}

	float atten = 1.0f;
	const float lightRadius = fmaxf(light.radius, 1.0f);
	const float lt = dist / lightRadius;
	if (lt >= 1.0f) {
		return 0.0f;
	}
	const float window = 1.0f - lt * lt;
	atten = window * window;

	if (light.projectedTextureIndex >= 0) {
		float s, t, f;
		if (!SampleLightProjectionUV(light, pos, s, t, f)) {
			return 0.0f;
		}
		if (textures) {
			float tex[4];
			SampleTextureSrgb(textures, light.projectedTextureIndex, s, t, tex);
			filter[0] *= tex[0];
			filter[1] *= tex[1];
			filter[2] *= tex[2];
		}
		const float ff = fmaxf(0.0f, 1.0f - f);
		atten *= ff * ff;
	} else if (light.coneAngle > 0.0f) {
		const float toSurface[3] = { -toLight[0], -toLight[1], -toLight[2] };
		atten *= SpotlightAttenuation(light, toSurface);
	} else if (light.textureIndex >= 0 && textures) {
		float tex[4];
		SampleTextureSrgb(textures, light.textureIndex, 0.5f, 0.5f, tex);
		filter[0] *= fminf(tex[0], 1.0f);
		filter[1] *= fminf(tex[1], 1.0f);
		filter[2] *= fminf(tex[2], 1.0f);
	}
	return atten;
}

/*
=================================================================================
LightImportance

Cheap, texture free estimate of a light's contribution at a surface point,
used to pick which lights get shadow rays. Zero only where the light can
not contribute at all, so importance sampling stays unbiased.
=================================================================================
*/
__device__ float LightImportance(const ptLight_t& light, const float* pos, const float* normal) {
	const float power = (0.2126f * light.color[0] + 0.7152f * light.color[1] + 0.0722f * light.color[2]) * light.intensity;
	if (power <= 0.0f) {
		return 0.0f;
	}

	if (light.type == PT_LIGHT_DIRECTIONAL) {
		return power * (fmaxf(-dot3(normal, light.direction), 0.0f) + 0.02f);
	}

	float d[3] = { light.position[0] - pos[0], light.position[1] - pos[1], light.position[2] - pos[2] };
	const float dist = length3(d);
	const float ndotl = dist > 1e-4f ? dot3(normal, d) / dist : 1.0f;
	// soft lights may still reach slightly back facing points
	const float cosTerm = fmaxf(ndotl, 0.0f) + 0.02f;

	float shape;
	if (light.boxFalloff) {
		const float (*lp)[4] = light.lightProject;
		const float q = lp[2][0] * pos[0] + lp[2][1] * pos[1] + lp[2][2] * pos[2] + lp[2][3];
		if (q <= 0.0001f) return 0.0f;
		const float s = (lp[0][0] * pos[0] + lp[0][1] * pos[1] + lp[0][2] * pos[2] + lp[0][3]) / q;
		const float t = (lp[1][0] * pos[0] + lp[1][1] * pos[1] + lp[1][2] * pos[2] + lp[1][3]) / q;
		const float f = lp[3][0] * pos[0] + lp[3][1] * pos[1] + lp[3][2] * pos[2] + lp[3][3];
		if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f || f < 0.0f || f > 1.0f) return 0.0f;
		const float ds = 2.0f * s - 1.0f, dt = 2.0f * t - 1.0f;
		if (light.coneAngle > 0.0f) {
			shape = 1.0f - (ds * ds + dt * dt) * 0.5f;
		} else {
			const float df = 2.0f * f - 1.0f;
			shape = 1.0f - (ds * ds + dt * dt + df * df) * (1.0f / 3.0f);
		}
		shape = fmaxf(shape, 0.05f);
	} else {
		const float lt = dist / fmaxf(light.radius, 1.0f);
		if (lt >= 1.0f) return 0.0f;
		shape = (1.0f - lt * lt);
		shape = fmaxf(shape * shape, 0.01f);
		if (light.projectedTextureIndex < 0 && light.coneAngle > 0.0f && dist > 1e-4f) {
			const float toSurface[3] = { -d[0] / dist, -d[1] / dist, -d[2] / dist };
			shape *= SpotlightAttenuation(light, toSurface);
		}
	}
	return power * cosTerm * shape;
}

/*
=================================================================================
PT_AreaLightSampledFrom

True when the PT_LIGHT_AREA built from this emissive triangle (see
idPathTracerScene::AddEmissiveAreaLight) has a nonzero LightImportance at pos,
i.e. light sampling at pos already accounts for the triangle's emission.
=================================================================================
*/
__device__ bool PT_AreaLightSampledFrom(const ptVertex_t* vertices, const ptTriangle_t& tri, const float* pos) {
	const float* p0 = vertices[tri.vertexIndices[0]].position;
	const float* p1 = vertices[tri.vertexIndices[1]].position;
	const float* p2 = vertices[tri.vertexIndices[2]].position;
	const float e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
	const float e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
	float n[3];
	cross3(e1, e2, n);
	const float radius = sqrtf(0.5f * length3(n));
	const float d[3] = {
		(p0[0] + p1[0] + p2[0]) * (1.0f / 3.0f) - pos[0],
		(p0[1] + p1[1] + p2[1]) * (1.0f / 3.0f) - pos[1],
		(p0[2] + p1[2] + p2[2]) * (1.0f / 3.0f) - pos[2] };
	return length3(d) < fmaxf(radius, 1.0f);
}

/*
=================================================================================
PT_SelectLights

Picks numPicks lights (with replacement) proportionally to LightImportance,
one weighted reservoir per pick, in a single pass over the lights. invPdf
receives 1 / (pmf * numPicks) for each pick so that the sum of
contribution * invPdf is an unbiased estimate of the sum over all lights.
Returns the number of picks, 0 when no light reaches the point.
=================================================================================
*/
#define PT_MAX_LIGHT_PICKS	8

__device__ int PT_SelectLights(
	const ptLight_t* lights,
	int numLights,
	const float* pos,
	const float* normal,
	int numPicks,
	ptRng_t* rng,
	int* picked,
	float* invPdf
) {
	numPicks = min(numPicks, PT_MAX_LIGHT_PICKS);
	if (numLights <= 0 || numPicks <= 0) {
		return 0;
	}

	float pickedW[PT_MAX_LIGHT_PICKS];
	float total = 0.0f;
	for (int i = 0; i < numLights; i++) {
		const float w = LightImportance(lights[i], pos, normal);
		if (w <= 0.0f) {
			continue;
		}
		total += w;
		const float p = w / total;
		for (int j = 0; j < numPicks; j++) {
			if (PT_Rand(rng) < p) {
				picked[j] = i;
				pickedW[j] = w;
			}
		}
	}
	if (total <= 0.0f) {
		return 0;
	}
	for (int j = 0; j < numPicks; j++) {
		invPdf[j] = total / (pickedW[j] * (float)numPicks);
	}
	return numPicks;
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
	ptRng_t* rng,
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

	const float stepSize = marchLength / (float)steps;
	const float stepTransmit = expf(-density * 0.5f * stepSize);
	float transmittance = 1.0f;

	for (int step = 0; step < steps && transmittance >= 0.01f; step++, transmittance *= stepTransmit) {
		const float t = (step + PT_Rand(rng)) * stepSize;
		const float samplePos[3] = {
			rayOrigin[0] + rayDirection[0] * t,
			rayOrigin[1] + rayDirection[1] * t,
			rayOrigin[2] + rayDirection[2] * t
		};

		float lightDir[3];
		float lightDist;
		if (light.type == PT_LIGHT_DIRECTIONAL) {
			lightDir[0] = -light.direction[0];
			lightDir[1] = -light.direction[1];
			lightDir[2] = -light.direction[2];
			lightDist = 1e10f;
		} else {
			lightDir[0] = light.position[0] - samplePos[0];
			lightDir[1] = light.position[1] - samplePos[1];
			lightDir[2] = light.position[2] - samplePos[2];
			lightDist = length3(lightDir);
			if (lightDist < 1e-4f) continue;
			lightDir[0] /= lightDist;
			lightDir[1] /= lightDist;
			lightDir[2] /= lightDist;
		}

		float volColor[3];
		const float attenuation = PT_LightFilter(light, samplePos, lightDist, lightDir, textures, volColor);
		if (attenuation <= 0.0f) continue;

		ptRay_t shadowRay;
		shadowRay.origin[0] = samplePos[0];
		shadowRay.origin[1] = samplePos[1];
		shadowRay.origin[2] = samplePos[2];
		shadowRay.direction[0] = lightDir[0];
		shadowRay.direction[1] = lightDir[1];
		shadowRay.direction[2] = lightDir[2];
		shadowRay.tMin = g_rayOffset;
		shadowRay.tMax = lightDist - g_rayOffset;
		shadowRay.worldTMin = 0.0f;
		shadowRay.flags = 0;

		float shadow[3];
		if (!TraceShadow(shadowRay, vertices, triangles, nodes, triIndices, materials, textures, shadow)) continue;
		const float shadowAtten = (shadow[0] + shadow[1] + shadow[2]) * (1.0f / 3.0f);

		float distFalloff = 1.0f;
		if (volFalloff > 0.0f && marchLength > 1.0f) {
			distFalloff = powf(1.0f - t / marchLength, volFalloff);
		}

		const float cosTheta = -dot3(rayDirection, lightDir);
		const float phase = HenyeyGreenstein(cosTheta, g);

		float contribution = light.intensity * density * stepSize * attenuation *
							 phase * shadowAtten * transmittance * distFalloff;
		contribution = fminf(contribution, 1.0f);

		scattering[0] += volColor[0] * contribution;
		scattering[1] += volColor[1] * contribution;
		scattering[2] += volColor[2] * contribution;
	}
}

/*
=================================================================================
SampleDirectLight

Next event estimation towards one light. wi receives the direction of the
(jittered) shadow ray for BRDF evaluation. worldTMin is forwarded to the
shadow ray (PT_WEAPON_CLEARANCE when shading the view weapon).
=================================================================================
*/
__device__ void SampleDirectLight(
	const ptLight_t& light,
	const float* position,
	const float* normal,
	float worldTMin,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* nodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	ptRng_t* rng,
	float* radiance,
	float* wi
) {
	radiance[0] = radiance[1] = radiance[2] = 0.0f;

	float lightPos[3];
	JitterLightPosition(light, rng, lightPos);

	float distance;
	if (light.type == PT_LIGHT_DIRECTIONAL) {
		wi[0] = -light.direction[0];
		wi[1] = -light.direction[1];
		wi[2] = -light.direction[2];
		distance = 1e10f;
	} else {
		wi[0] = lightPos[0] - position[0];
		wi[1] = lightPos[1] - position[1];
		wi[2] = lightPos[2] - position[2];
		distance = length3(wi);
		if (distance < 1e-4f) return;
		wi[0] /= distance;
		wi[1] /= distance;
		wi[2] /= distance;
	}

	const float ndotl = dot3(normal, wi);
	if (ndotl <= 0.0f) return;

	float lightColor[3];
	const float attenuation = PT_LightFilter(light, position, distance, wi, textures, lightColor);
	if (attenuation <= 0.0f) return;

	float contribution = light.intensity * ndotl * attenuation;
	if (contribution > 10.0f) contribution = 10.0f;

	ptRay_t shadowRay;
	shadowRay.origin[0] = position[0] + normal[0] * g_rayOffset;
	shadowRay.origin[1] = position[1] + normal[1] * g_rayOffset;
	shadowRay.origin[2] = position[2] + normal[2] * g_rayOffset;
	shadowRay.direction[0] = wi[0];
	shadowRay.direction[1] = wi[1];
	shadowRay.direction[2] = wi[2];
	shadowRay.tMin = g_rayOffset;
	shadowRay.tMax = distance - g_rayOffset;
	shadowRay.worldTMin = worldTMin;
	shadowRay.flags = 0;

	float shadow[3];
	if (!TraceShadow(shadowRay, vertices, triangles, nodes, triIndices, materials, textures, shadow)) {
		return;
	}

	radiance[0] = lightColor[0] * contribution * shadow[0];
	radiance[1] = lightColor[1] * contribution * shadow[1];
	radiance[2] = lightColor[2] * contribution * shadow[2];
}

/*
=================================================================================
SampleIndirectLight - one diffuse bounce with direct lighting at the hit

Only used when r_ptMaxDepth is 1, deeper paths already continue the bounce.
=================================================================================
*/
__device__ void SampleIndirectLight(
	const float* position,
	const float* normal,
	const float* direction,
	float worldTMin,
	const ptVertex_t* vertices,
	const ptTriangle_t* triangles,
	const ptBVHNode_t* bvhNodes,
	const int* triIndices,
	const ptMaterial_t* materials,
	const cudaTexture_t* textures,
	const ptLight_t* lights,
	int numLights,
	ptRng_t* rng,
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
	ray.worldTMin = worldTMin;
	ray.flags = 0;

	ptHitInfo_t hitInfo;
	float albedo[4];
	float tu = 0.0f, tv = 0.0f;
	bool foundOpaque = false;

	const int MAX_INDIRECT_STEPS = 4;
	for (int step = 0; step < MAX_INDIRECT_STEPS; step++) {
		if (!IntersectBVH(ray, vertices, triangles, bvhNodes, triIndices, materials, hitInfo)) {
			return;
		}

		const ptMaterial_t& mat = materials[hitInfo.materialIndex];
		tu = hitInfo.texcoord[0];
		tv = hitInfo.texcoord[1];
		PT_ApplyTexTransform(mat, tu, tv);
		albedo[0] = albedo[1] = albedo[2] = albedo[3] = 1.0f;
		if (mat.albedoTexture >= 0 && textures) {
			SampleTextureSrgb(textures, mat.albedoTexture, tu, tv, albedo);
		}
		for (int c = 0; c < 4; c++) {
			albedo[c] *= mat.albedo[c];
		}

		const bool passThrough = mat.noShadows || mat.isAmbientOnly || mat.transmission > 0.5f
			|| mat.blendMode == PT_BLEND_ADD || (mat.alphaTest > 0.0f && albedo[3] < mat.alphaTest);
		if (!passThrough) {
			foundOpaque = true;
			break;
		}
		ray.origin[0] = hitInfo.position[0] + ray.direction[0] * g_rayOffset;
		ray.origin[1] = hitInfo.position[1] + ray.direction[1] * g_rayOffset;
		ray.origin[2] = hitInfo.position[2] + ray.direction[2] * g_rayOffset;
		ray.worldTMin = fmaxf(0.0f, ray.worldTMin - hitInfo.t);
	}
	if (!foundOpaque) {
		return;
	}

	const ptMaterial_t& material = materials[hitInfo.materialIndex];

	// skip emitters the caller's light sampling already covered
	if (!(hitInfo.triFlags & PT_TRI_AREA_LIGHT)
		|| !PT_AreaLightSampledFrom(vertices, triangles[hitInfo.triangleIndex], position)) {
		float glow[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
		if (material.emissionTexture >= 0 && textures) {
			SampleTextureSrgb(textures, material.emissionTexture, tu, tv, glow);
		}
		radiance[0] += material.emission[0] * glow[0];
		radiance[1] += material.emission[1] * glow[1];
		radiance[2] += material.emission[2] * glow[2];
	}

	int pick;
	float invPdf;
	if (PT_SelectLights(lights, numLights, hitInfo.position, hitInfo.normal, 1, rng, &pick, &invPdf) > 0) {
		float directLight[3], wi[3];
		SampleDirectLight(lights[pick], hitInfo.position, hitInfo.normal, 0.0f,
			vertices, triangles, bvhNodes, triIndices, materials, textures, rng, directLight, wi);

		radiance[0] += albedo[0] * directLight[0] * invPdf / PT_PI;
		radiance[1] += albedo[1] * directLight[1] * invPdf / PT_PI;
		radiance[2] += albedo[2] * directLight[2] * invPdf / PT_PI;
	}

	// clamp indirect
	const float maxIndirect = 1.5f;
	radiance[0] = fminf(radiance[0], maxIndirect);
	radiance[1] = fminf(radiance[1], maxIndirect);
	radiance[2] = fminf(radiance[2], maxIndirect);
}

#endif // __CU_RENDERER_SAMPLING_CUH__
