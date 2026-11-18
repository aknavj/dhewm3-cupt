#ifndef __PT_CUDA_PHYSICAL_CUH__
#define __PT_CUDA_PHYSICAL_CUH__

#ifdef PT_PHYSICAL

#include "renderer_pt/pt_physical.h"
#include "renderer_pt/cuda/pt_cuda_math.cuh"

__device__ inline float PT_PhysicalF0(float albedo, float metallic, float ior) {
	const float f = (ior - 1.0f) / (ior + 1.0f);
	return f * f * (1.0f - metallic) + albedo * metallic;
}

__device__ inline float PT_PhysicalGGX(float nh, float roughness) {
	const float alpha = roughness * roughness;
	const float a2 = alpha * alpha;
	// This form avoids cancellation at nh=1 for polished surfaces.
	const float d = (1.0f - nh * nh) + nh * nh * a2;
	return a2 / fmaxf(PT_PI * d * d, 1e-20f);
}

__device__ inline void PT_EvaluatePhysicalMaterial(
	const float* albedo, const float* n, const float* wo, const float* wi,
	float metallic, float roughness, float ior, float* brdf, float* diffuse
) {
	const float nl = fmaxf(dot3(n, wi), 0.0f), nv = fmaxf(dot3(n, wo), 0.0f);
	float h[3] = { wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2] };
	normalize3(h);
	const float nh = fmaxf(dot3(n, h), 0.0f), vh = fmaxf(dot3(wo, h), 0.0f);
	const float alpha = roughness * roughness, a2 = alpha * alpha;
	const float visibility = 0.5f / fmaxf(
		nl * sqrtf(nv * nv * (1.0f - a2) + a2)
		+ nv * sqrtf(nl * nl * (1.0f - a2) + a2), 1e-12f);
	const float fresnelWeight = powf(1.0f - vh, 5.0f);
	for (int c = 0; c < 3; c++) {
		const float f0 = PT_PhysicalF0(albedo[c], metallic, ior);
		const float f = f0 + (1.0f - f0) * fresnelWeight;
		const float transmitView = (1.0f - f0) * (1.0f - powf(1.0f - nv, 5.0f));
		const float transmitLight = (1.0f - f0) * (1.0f - powf(1.0f - nl, 5.0f));
		const float d = transmitView * transmitLight * (1.0f - metallic) * albedo[c] / PT_PI;
		brdf[c] = nl > 0.0f && nv > 0.0f ? d + PT_PhysicalGGX(nh, roughness) * visibility * f : 0.0f;
		if (diffuse) diffuse[c] = nl > 0.0f && nv > 0.0f ? d : 0.0f;
	}
}

__device__ inline float PT_PhysicalSpecularProbability(const float* albedo, float metallic, float ior) {
	float f0 = 0.0f;
	for (int c = 0; c < 3; c++) f0 = fmaxf(f0, PT_PhysicalF0(albedo[c], metallic, ior));
	return clamp(f0, 0.05f, 0.95f);
}

__device__ inline float PT_PhysicalBsdfPdf(
	const float* n, const float* wo, const float* wi, float roughness, float specularProbability
) {
	const float nl = dot3(n, wi);
	if (nl <= 0.0f) return 0.0f;
	float h[3] = { wo[0] + wi[0], wo[1] + wi[1], wo[2] + wi[2] };
	normalize3(h);
	const float nh = fmaxf(dot3(n, h), 0.0f), vh = fmaxf(dot3(wo, h), 0.0f);
	const float specularPdf = PT_PhysicalGGX(nh, roughness) * nh / fmaxf(4.0f * vh, 1e-12f);
	return (1.0f - specularProbability) * nl / PT_PI + specularProbability * specularPdf;
}

__device__ inline void PT_PhysicalLens(
	const ptRenderParams_t& p, ptRng_t* rng, float* origin, float* direction
) {
	if (p.physicalApertureRadius <= 0.0f) return;
	const float radius = sqrtf(PT_Rand(rng)) * p.physicalApertureRadius;
	const float angle = 2.0f * PT_PI * PT_Rand(rng);
	const float x = radius * cosf(angle), y = radius * sinf(angle);
	const float distance = p.physicalFocusDistance / fmaxf(dot3(direction, p.camForward), 1e-6f);
	for (int c = 0; c < 3; c++) {
		const float offset = x * p.camRight[c] + y * p.camUp[c];
		origin[c] += offset;
		direction[c] = direction[c] * distance - offset;
	}
	normalize3(direction);
}

#endif
#endif
