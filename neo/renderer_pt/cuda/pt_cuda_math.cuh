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
hashRNG - PCG-style hash mixing for RNG
========================
*/
__device__ inline unsigned int hashRNG(unsigned int seed) {
	seed = (seed ^ 61u) ^ (seed >> 16u);
	seed *= 9u;
	seed = seed ^ (seed >> 4u);
	seed *= 0x27d4eb2du;
	seed = seed ^ (seed >> 15u);
	return seed;
}

/*
========================
hashRandFloat - Advance RNG state and return a float in [0, 1)
========================
*/
__device__ inline float hashRandFloat(unsigned int& state) {
	state = hashRNG(state);
	return (float)(state & 0x00FFFFFF) / (float)0x01000000;
}

// legacy aliases for backward compatibility with old kernels
__device__ __forceinline__ unsigned int cupt_hash(unsigned int seed) {
	return hashRNG(seed);
}

__device__ __forceinline__ float cupt_random(unsigned int seed) {
	return float(hashRNG(seed)) / float(0xFFFFFFFFu);
}

#endif // HAVE_CUDA
#endif // __CU_RENDERER_MATH_CUH__