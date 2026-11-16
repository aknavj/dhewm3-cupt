#ifndef __PT_KERNEL_DENOISING_CUH__
#define __PT_KERNEL_DENOISING_CUH__

#include "renderer_pt/cuda/pt_cuda_device.h"

__device__ inline float PT_DenoiseLuminance(const float *color) {
	return color[0] * 0.2126f + color[1] * 0.7152f + color[2] * 0.0722f;
}

__device__ inline float PT_DenoiseNormalDot(const ptCudaDenoiseGuide_t &a, const ptCudaDenoiseGuide_t &b) {
	return a.normal[0] * b.normal[0] + a.normal[1] * b.normal[1] + a.normal[2] * b.normal[2];
}

__device__ inline bool PT_DenoiseSameSurface(const ptCudaDenoiseGuide_t &a, const ptCudaDenoiseGuide_t &b) {
	return b.depth > 0.0f && fabsf(a.depth - b.depth) < fmaxf(0.01f, a.depth * 0.02f)
		&& PT_DenoiseNormalDot(a, b) > 0.9f;
}

__device__ inline void PT_DenoiseIllumination(const float *radiance, const ptCudaDenoiseGuide_t &guide, float *color) {
	for (int c = 0; c < 3; c++) color[c] = radiance[c] / guide.albedo[c];
}

__global__ void SVGFTemporalKernel(
	const float *radiance, const ptCudaDenoiseGuide_t *guides, const ptCudaDenoiseGuide_t *previousGuides,
	const float *previousHistory, const float *previousMoments, float *output, float *moments,
	ptRenderParams_t params, ptRenderParams_t previousParams, int historyValid
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= params.width || y >= params.height) return;
	const int pixel = y * params.width + x;
	const int index = pixel * 4;
	const ptCudaDenoiseGuide_t &guide = guides[pixel];
	if (guide.depth <= 0.0f) {
		for (int c = 0; c < 3; c++) output[index + c] = radiance[index + c];
		output[index + 3] = 0.0f;
		for (int c = 0; c < 4; c++) moments[index + c] = 0.0f;
		return;
	}

	float current[3];
	PT_DenoiseIllumination(radiance + index, guide, current);
	const float luminance = PT_DenoiseLuminance(current);
	float localMin[3] = { current[0], current[1], current[2] };
	float localMax[3] = { current[0], current[1], current[2] };
	float localMean = 0.0f, localSquare = 0.0f, localCount = 0.0f;
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			const int sx = x + dx, sy = y + dy;
			if (sx < 0 || sx >= params.width || sy < 0 || sy >= params.height) continue;
			const int neighbor = sy * params.width + sx;
			if (!PT_DenoiseSameSurface(guide, guides[neighbor])) continue;
			float color[3];
			PT_DenoiseIllumination(radiance + neighbor * 4, guides[neighbor], color);
			const float lum = PT_DenoiseLuminance(color);
			localMean += lum;
			localSquare += lum * lum;
			localCount += 1.0f;
			for (int c = 0; c < 3; c++) {
				localMin[c] = fminf(localMin[c], color[c]);
				localMax[c] = fmaxf(localMax[c], color[c]);
			}
		}
	}
	localMean /= fmaxf(localCount, 1.0f);
	const float localVariance = fmaxf(0.0f, localSquare / fmaxf(localCount, 1.0f) - localMean * localMean);

	float history[3] = {};
	float historyMoments[3] = {};
	float totalWeight = 0.0f;
	if (historyValid && guide.historyValid) {
		float delta[3], previousDepth = 0.0f, horizontal = 0.0f, vertical = 0.0f;
		for (int c = 0; c < 3; c++) {
			delta[c] = guide.position[c] - previousParams.camPos[c];
			previousDepth += delta[c] * previousParams.camForward[c];
			horizontal += delta[c] * previousParams.camRight[c];
			vertical += delta[c] * previousParams.camUp[c];
		}
		const float tanX = tanf(previousParams.fovX * PT_PI / 360.0f);
		const float tanY = tanf(previousParams.fovY * PT_PI / 360.0f);
		if (previousDepth > 0.0f && tanX > 1e-6f && tanY > 1e-6f) {
			const float px = (horizontal / (previousDepth * tanX) + 1.0f) * 0.5f * params.width - 0.5f;
			const float py = (vertical / (previousDepth * tanY) + 1.0f) * 0.5f * params.height - 0.5f;
			if (px > -1.0f && px < params.width && py > -1.0f && py < params.height) {
				const int ix = (int)floorf(px), iy = (int)floorf(py);
				const float fx = px - ix, fy = py - iy;
				for (int dy = 0; dy < 2; dy++) {
					for (int dx = 0; dx < 2; dx++) {
						const int sx = ix + dx, sy = iy + dy;
						if (sx < 0 || sx >= params.width || sy < 0 || sy >= params.height) continue;
						const int neighbor = sy * params.width + sx;
						const ptCudaDenoiseGuide_t &oldGuide = previousGuides[neighbor];
						if (!oldGuide.historyValid || oldGuide.depth <= 0.0f
							|| oldGuide.materialIndex != guide.materialIndex
							|| fabsf(oldGuide.depth - previousDepth) > fmaxf(0.01f, previousDepth * 0.02f)
							|| PT_DenoiseNormalDot(guide, oldGuide) < 0.9f) continue;
						bool sameAlbedo = true;
						for (int c = 0; c < 3; c++) {
							sameAlbedo &= fabsf(guide.albedo[c] - oldGuide.albedo[c])
								<= 0.02f + 0.2f * fmaxf(guide.albedo[c], oldGuide.albedo[c]);
						}
						if (!sameAlbedo || previousMoments[neighbor * 4 + 2] < 1.0f) continue;
						const float weight = (dx ? fx : 1.0f - fx) * (dy ? fy : 1.0f - fy);
						for (int c = 0; c < 3; c++) {
							history[c] += previousHistory[neighbor * 4 + c] * weight;
							historyMoments[c] += previousMoments[neighbor * 4 + c] * weight;
						}
						totalWeight += weight;
					}
				}
			}
		}
	}

	float historyLength = 1.0f;
	float mean = luminance, square = luminance * luminance;
	if (totalWeight > 0.001f) {
		for (int c = 0; c < 3; c++) {
			history[c] /= totalWeight;
			historyMoments[c] /= totalWeight;
		}
		// Neighborhood-based change detection is conservative, not A-SVGF's correlated gradient estimator.
		const float oldVariance = fmaxf(0.0f, historyMoments[1] - historyMoments[0] * historyMoments[0]);
		const float changeLimit = 3.0f * sqrtf(localVariance + oldVariance)
			+ 0.1f * fmaxf(0.1f, localMean);
		if (fabsf(localMean - historyMoments[0]) <= changeLimit) {
			historyLength = fminf(32.0f, historyMoments[2] + 1.0f);
			const float alpha = fmaxf(0.08f, 1.0f / historyLength);
			const float momentAlpha = fmaxf(0.1f, 1.0f / historyLength);
			mean += (historyMoments[0] - mean) * (1.0f - momentAlpha);
			square += (historyMoments[1] - square) * (1.0f - momentAlpha);
			for (int c = 0; c < 3; c++) {
				const float margin = 0.25f * (localMax[c] - localMin[c]);
				const float clipped = fminf(localMax[c] + margin, fmaxf(localMin[c] - margin, history[c]));
				current[c] += (clipped - current[c]) * (1.0f - alpha);
			}
		}
	}
	float variance = fmaxf(0.0f, square - mean * mean);
	if (historyLength < 4.0f) variance = fmaxf(variance, localVariance) * (4.0f / historyLength);
	for (int c = 0; c < 3; c++) output[index + c] = current[c];
	output[index + 3] = variance;
	moments[index] = mean;
	moments[index + 1] = square;
	moments[index + 2] = historyLength;
	moments[index + 3] = 0.0f;
}

__global__ void SVGFAtrousKernel(
	const float *input, const ptCudaDenoiseGuide_t *guides, float *output,
	int width, int height, int step, float strength
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	const int pixel = y * width + x;
	const int index = pixel * 4;
	const ptCudaDenoiseGuide_t &guide = guides[pixel];
	if (guide.depth <= 0.0f) {
		for (int c = 0; c < 4; c++) output[index + c] = input[index + c];
		return;
	}

	float variance = 0.0f, varianceWeight = 0.0f, depthSlope = 0.0f;
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			const int sx = x + dx, sy = y + dy;
			if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
			const int neighbor = sy * width + sx;
			if (!PT_DenoiseSameSurface(guide, guides[neighbor])) continue;
			const float weight = (dx == 0 ? 2.0f : 1.0f) * (dy == 0 ? 2.0f : 1.0f);
			variance += input[neighbor * 4 + 3] * weight;
			varianceWeight += weight;
			depthSlope = fmaxf(depthSlope, fabsf(guide.depth - guides[neighbor].depth));
		}
	}
	variance /= fmaxf(1.0f, varianceWeight);
	const float luminance = PT_DenoiseLuminance(input + index);
	const float colorSigma = fmaxf(0.0001f, 8.0f * strength * sqrtf(fmaxf(0.0f, variance)));
	float sum[3] = {};
	float sumVariance = 0.0f, totalWeight = 0.0f;
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			const int sx = x + dx * step, sy = y + dy * step;
			if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
			const int neighbor = sy * width + sx;
			const ptCudaDenoiseGuide_t &sampleGuide = guides[neighbor];
			if (sampleGuide.depth <= 0.0f) continue;
			const float depthSigma = fmaxf(0.01f, guide.depth * 0.002f)
				+ depthSlope * step * sqrtf((float)(dx * dx + dy * dy));
			const float colorDelta = fabsf(luminance - PT_DenoiseLuminance(input + neighbor * 4));
			const float normalDot = fminf(1.0f, fmaxf(0.0f, PT_DenoiseNormalDot(guide, sampleGuide)));
			const float normalWeight = dx == 0 && dy == 0 ? 1.0f : powf(normalDot, 32.0f);
			const float weight = (dx == 0 ? 2.0f : 1.0f) * (dy == 0 ? 2.0f : 1.0f)
				* normalWeight
				* expf(-fabsf(guide.depth - sampleGuide.depth) / depthSigma - colorDelta / colorSigma);
			for (int c = 0; c < 3; c++) sum[c] += input[neighbor * 4 + c] * weight;
			sumVariance += input[neighbor * 4 + 3] * weight * weight;
			totalWeight += weight;
		}
	}
	// The center tap always has weight four, including at image boundaries.
	for (int c = 0; c < 3; c++) output[index + c] = sum[c] / totalWeight;
	output[index + 3] = sumVariance / (totalWeight * totalWeight);
}

__global__ void SVGFResolveKernel(
	const float *illumination, const float *accumulation, const ptCudaDenoiseGuide_t *guides,
	float *output, int width, int height
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	const int pixel = y * width + x;
	const int index = pixel * 4;
	for (int c = 0; c < 3; c++) {
		output[index + c] = guides[pixel].depth > 0.0f
			? illumination[index + c] * guides[pixel].albedo[c] : accumulation[index + c];
	}
	output[index + 3] = accumulation[index + 3];
}

#endif /* !__PT_KERNEL_DENOISING_CUH__ */
