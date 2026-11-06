#ifndef __CU_RENDERER_MATH_CUH__
#define __CU_RENDERER_MATH_CUH__

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

/*
========================
clamp
========================
*/
__device__ inline float clamp(float x, float a, float b) {
	return fmaxf(a, fminf(b, x));
}

/*
========================
dot3
========================
*/
__device__ inline float dot3(const float* a, const float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/*
========================
cross3
========================
*/
__device__ inline void cross3(const float* a, const float* b, float* result) {
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = a[2] * b[0] - a[0] * b[2];
	result[2] = a[0] * b[1] - a[1] * b[0];
}

/*
========================
normalize3
========================
*/
__device__ inline void normalize3(float* v) {
	float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 1e-6f) {
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
}

/*
========================
length3
========================
*/
__device__ inline float length3(const float* v) {
	return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/*
========================
PCG random numbers

pcg_hash (Jarzynski & Olano, "Hash Functions for GPU Rendering") seeds one
32 bit LCG state per pixel and frame; every draw advances the LCG and runs
the PCG RXS-M-XS output permutation. 4 bytes of state instead of curand's
48 byte XORWOW state and no expensive curand_init.
========================
*/
struct ptRng_t {
	unsigned int state;
};

__device__ __forceinline__ unsigned int PT_PcgPermute(unsigned int state) {
	unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __forceinline__ unsigned int PT_PcgHash(unsigned int v) {
	return PT_PcgPermute(v * 747796405u + 2891336453u);
}

__device__ __forceinline__ void PT_RngInit(ptRng_t* rng, unsigned int pixel, unsigned int frame) {
	rng->state = PT_PcgHash(pixel ^ PT_PcgHash(frame + 0x9E3779B9u));
}

// uniform float in [0, 1)
__device__ __forceinline__ float PT_Rand(ptRng_t* rng) {
	rng->state = rng->state * 747796405u + 2891336453u;
	return (float)(PT_PcgPermute(rng->state) >> 8) * (1.0f / 16777216.0f);
}

// stateless hash, used for per-triangle debug colors
__device__ __forceinline__ unsigned int cupt_hash(unsigned int seed) {
	return PT_PcgHash(seed);
}

/*
========================
PT_SrgbToLinear

Plain 2.2 power curve: it is the exact inverse of the default r_ptGamma, so
an unlit texture (GUIs, ambient-only overlays) round-trips unchanged and lit
surfaces reproduce the GL renderer's gamma-space modulation.
========================
*/
__device__ __forceinline__ float PT_SrgbToLinear(float c) {
	return c > 0.0f ? exp2f(2.2f * log2f(c)) : 0.0f;
}

#endif // HAVE_CUDA
#endif // __CU_RENDERER_MATH_CUH__