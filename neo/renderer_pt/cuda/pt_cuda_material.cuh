#ifndef __CU_RENDERER_MATERIAL_CUH__
#define __CU_RENDERER_MATERIAL_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"

/*
=================================================================================
Texture Sampling (bilinear filtering)
=================================================================================
*/
__device__ inline void SampleTexture(const cudaTexture_t* textures, int texIndex, float u, float v, float* color) {
	if (texIndex < 0 || texIndex >= MAX_TEXTURES || !textures[texIndex].data
		|| textures[texIndex].width <= 0 || textures[texIndex].height <= 0) {
		color[0] = color[1] = color[2] = color[3] = 1.0f;
		return;
	}

	const cudaTexture_t& tex = textures[texIndex];

	// wrap texture coordinates
	u = u - floorf(u);
	v = v - floorf(v);

	// convert to texel coordinates
	float x = u * (float)tex.width;
	float y = v * (float)tex.height;

	int x0 = (int)x % tex.width;
	int y0 = (int)y % tex.height;
	int x1 = (x0 + 1) % tex.width;
	int y1 = (y0 + 1) % tex.height;

	float fx = x - floorf(x);
	float fy = y - floorf(y);

	// bilinear interpolation
	int idx00 = (y0 * tex.width + x0) * 4;
	int idx10 = (y0 * tex.width + x1) * 4;
	int idx01 = (y1 * tex.width + x0) * 4;
	int idx11 = (y1 * tex.width + x1) * 4;

	for (int c = 0; c < 4; c++) {
		float v0 = tex.data[idx00 + c] * (1.0f - fx) + tex.data[idx10 + c] * fx;
		float v1 = tex.data[idx01 + c] * (1.0f - fx) + tex.data[idx11 + c] * fx;
		color[c] = (v0 * (1.0f - fy) + v1 * fy) / 255.0f;
	}
}

/*
=================================================================================
PBR Material Evaluation - Cook-Torrance BRDF
=================================================================================
*/

// Fresnel-Schlick approximation
__device__ inline float FresnelSchlick(float cosTheta, float F0) {
	float x = 1.0f - cosTheta;
	float x2 = x * x;
	return F0 + (1.0f - F0) * x2 * x2 * x;  // x^5
}

__device__ inline void FresnelSchlickVec3(float cosTheta, const float* F0, float* result) {
	float x = 1.0f - cosTheta;
	float x2 = x * x;
	float x5 = x2 * x2 * x;
	result[0] = F0[0] + (1.0f - F0[0]) * x5;
	result[1] = F0[1] + (1.0f - F0[1]) * x5;
	result[2] = F0[2] + (1.0f - F0[2]) * x5;
}

// GGX normal distribution function
__device__ inline float DistributionGGX(float NdotH, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH2 = NdotH * NdotH;

	float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
	denom = PT_PI * denom * denom;

	return a2 / fmaxf(denom, 0.0001f);
}

// Smith's geometry shadowing function with GGX
__device__ inline float GeometrySchlickGGX(float NdotV, float roughness) {
	float r = roughness + 1.0f;
	float k = (r * r) / 8.0f;

	return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ inline float GeometrySmith(float NdotV, float NdotL, float roughness) {
	float ggx1 = GeometrySchlickGGX(NdotV, roughness);
	float ggx2 = GeometrySchlickGGX(NdotL, roughness);
	return ggx1 * ggx2;
}

// simple Lambertian BRDF (backward compatibility)
__device__ void EvaluateMaterial(
	const float* albedo,
	const float* normal,
	const float* wo,
	const float* wi,
	float* brdf
) {
	float ndotwi = fmaxf(0.0f, dot3(normal, wi));
	float diffuse = ndotwi / PT_PI;

	brdf[0] = albedo[0] * diffuse;
	brdf[1] = albedo[1] * diffuse;
	brdf[2] = albedo[2] * diffuse;
}

// full PBR material evaluation with metallic-roughness workflow
__device__ void EvaluatePBRMaterial(
	const float* albedo,
	const float* normal,
	const float* wo,      // view direction (from surface to camera)
	const float* wi,      // light direction (from surface to light)
	float metallic,
	float roughness,
	float specularBoost,  // Doom 3 interaction shader doubles specular
	const float* specColor,  // specular tint from specular map
	float* brdf
) {
	float NdotL = fmaxf(0.0f, dot3(normal, wi));
	float NdotV = fmaxf(0.0f, dot3(normal, wo));

	if (NdotL <= 0.0f || NdotV <= 0.0f) {
		brdf[0] = brdf[1] = brdf[2] = 0.0f;
		return;
	}

	// half vector for specular
	float h[3] = {
		wo[0] + wi[0],
		wo[1] + wi[1],
		wo[2] + wi[2]
	};
	normalize3(h);

	float NdotH = fmaxf(0.0f, dot3(normal, h));
	float HdotV = fmaxf(0.0f, dot3(h, wo));

	// clamp roughness to prevent numerical issues
	roughness = fmaxf(0.04f, roughness);

	// calculate F0 (surface reflection at zero incidence)
	float F0[3];
	F0[0] = 0.04f * (1.0f - metallic) + albedo[0] * metallic;
	F0[1] = 0.04f * (1.0f - metallic) + albedo[1] * metallic;
	F0[2] = 0.04f * (1.0f - metallic) + albedo[2] * metallic;

	// Cook-Torrance specular BRDF: D * G * F / (4 * NdotV * NdotL)
	float D = DistributionGGX(NdotH, roughness);
	float G = GeometrySmith(NdotV, NdotL, roughness);
	float F[3];
	FresnelSchlickVec3(HdotV, F0, F);

	float denominator = 4.0f * NdotV * NdotL;
	float specular[3];
	specular[0] = (D * G * F[0]) / fmaxf(denominator, 0.001f) * specularBoost * specColor[0];
	specular[1] = (D * G * F[1]) / fmaxf(denominator, 0.001f) * specularBoost * specColor[1];
	specular[2] = (D * G * F[2]) / fmaxf(denominator, 0.001f) * specularBoost * specColor[2];

	// diffuse contribution (energy conserving)
	// kD = (1 - F) * (1 - metallic) — metals have no diffuse
	float kD[3];
	kD[0] = (1.0f - F[0]) * (1.0f - metallic);
	kD[1] = (1.0f - F[1]) * (1.0f - metallic);
	kD[2] = (1.0f - F[2]) * (1.0f - metallic);

	// Lambertian diffuse: albedo / pi
	float diff[3];
	diff[0] = kD[0] * albedo[0] / PT_PI;
	diff[1] = kD[1] * albedo[1] / PT_PI;
	diff[2] = kD[2] * albedo[2] / PT_PI;

	// return pure BRDF f(wo,wi) = diffuse + specular (NO NdotL here)
	brdf[0] = diff[0] + specular[0];
	brdf[1] = diff[1] + specular[1];
	brdf[2] = diff[2] + specular[2];
}

#endif // __CU_RENDERER_MATERIAL_CUH__