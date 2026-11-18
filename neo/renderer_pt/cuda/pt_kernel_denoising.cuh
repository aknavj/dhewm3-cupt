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

__device__ inline const float *PT_DenoiseReflectance(const ptCudaDenoiseGuide_t &guide, int signal) {
	return signal == 0 ? guide.albedo : guide.specularAlbedo;
}

__device__ inline void PT_DenoiseIllumination(const float *radiance, const ptCudaDenoiseGuide_t &guide, float *color, int signal) {
	const float *reflectance = PT_DenoiseReflectance(guide, signal);
	for (int c = 0; c < 3; c++) color[c] = radiance[c] / reflectance[c];
}

__global__ void SVGFCaptureGuidesKernel(
	ptCudaDenoiseGuide_t *guides, const ptCudaDenoiseGuide_t *previousGuides, ptRenderParams_t params
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= params.width || y >= params.height || params.frameIndex == 0) return;
	const int pixel = y * params.width + x;
	ptCudaDenoiseGuide_t &guide = guides[pixel];
	const ptCudaDenoiseGuide_t &old = previousGuides[pixel];
	float planeDistance = 0.0f;
	for (int c = 0; c < 3; c++) planeDistance += (guide.position[c] - old.position[c]) * old.normal[c];
	if (guide.depth <= 0.0f || old.depth <= 0.0f || guide.materialIndex != old.materialIndex
		|| PT_DenoiseNormalDot(guide, old) < 0.9f || fabsf(planeDistance) > fmaxf(0.01f, old.depth * 0.002f)) {
		if (guide.depth > 0.0f || old.depth > 0.0f || guide.coverage != old.coverage) guide.coverage = PT_DENOISE_MIXED;
		guide.depth = 0.0f;
		guide.historyValid = 0;
		return;
	}
	const float alpha = 1.0f / (params.frameIndex + 1.0f);
	for (int c = 0; c < 3; c++) {
		guide.position[c] = old.position[c] + (guide.position[c] - old.position[c]) * alpha;
		guide.albedo[c] = old.albedo[c] + (guide.albedo[c] - old.albedo[c]) * alpha;
		guide.specularAlbedo[c] = old.specularAlbedo[c] + (guide.specularAlbedo[c] - old.specularAlbedo[c]) * alpha;
		guide.shadingNormal[c] = old.shadingNormal[c] + (guide.shadingNormal[c] - old.shadingNormal[c]) * alpha;
		guide.normal[c] = old.normal[c];
	}
	float normalLength = 0.0f;
	for (int c = 0; c < 3; c++) normalLength += guide.shadingNormal[c] * guide.shadingNormal[c];
	if (normalLength > 1e-12f) for (int c = 0; c < 3; c++) guide.shadingNormal[c] /= sqrtf(normalLength);
	guide.depth = old.depth + (guide.depth - old.depth) * alpha;
	guide.roughness = old.roughness + (guide.roughness - old.roughness) * alpha;
}

__global__ void SVGFCaptureKernel(
	const float *radiance, const ptCudaDenoiseGuide_t *guides, const float *previousMean,
	float *mean, float *output, float *moments, ptRenderParams_t params, int signal
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= params.width || y >= params.height) return;
	const int pixel = y * params.width + x, index = pixel * 4;
	const float count = params.frameIndex + 1.0f, alpha = 1.0f / count;
	const float sampleLum = PT_DenoiseLuminance(radiance + index);
	for (int c = 0; c < 3; c++) {
		mean[index + c] = params.frameIndex == 0 ? radiance[index + c]
			: previousMean[index + c] + (radiance[index + c] - previousMean[index + c]) * alpha;
	}
	mean[index + 3] = params.frameIndex == 0 ? sampleLum * sampleLum
		: previousMean[index + 3] + (sampleLum * sampleLum - previousMean[index + 3]) * alpha;
	const ptCudaDenoiseGuide_t &guide = guides[pixel];
	if (guide.depth <= 0.0f) {
		for (int c = 0; c < 3; c++) output[index + c] = mean[index + c];
		output[index + 3] = 0.0f;
		moments[index] = moments[index + 1] = moments[index + 2] = 0.0f;
		moments[index + 3] = PT_DENOISE_UNSUPPORTED;
		return;
	}
	PT_DenoiseIllumination(mean + index, guide, output + index, signal);
	const float rawMean = PT_DenoiseLuminance(mean + index);
	const float reflectance = fmaxf(0.01f, PT_DenoiseLuminance(PT_DenoiseReflectance(guide, signal)));
	const float variance = fmaxf(0.0f, mean[index + 3] - rawMean * rawMean)
		/ (fmaxf(1.0f, count - 1.0f) * reflectance * reflectance);
	output[index + 3] = variance;
	const float luminance = PT_DenoiseLuminance(output + index);
	moments[index] = luminance;
	moments[index + 1] = luminance * luminance + variance;
	moments[index + 2] = count;
	moments[index + 3] = params.frameIndex == 0 ? PT_DENOISE_RESET : PT_DENOISE_REUSED;
}

__global__ void SVGFTemporalKernel(
	const float *radiance, const ptCudaDenoiseGuide_t *guides, const ptCudaDenoiseGuide_t *previousGuides,
	const float *previousHistory, const float *previousMoments, float *output, float *moments,
	ptRenderParams_t params, ptRenderParams_t previousParams, int historyValid, int signal
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
		moments[index + 3] = PT_DENOISE_UNSUPPORTED;
		return;
	}

	float current[3];
	PT_DenoiseIllumination(radiance + index, guide, current, signal);
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
			PT_DenoiseIllumination(radiance + neighbor * 4, guides[neighbor], color, signal);
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
	int rejection = !historyValid ? PT_DENOISE_RESET : (!guide.historyValid ? PT_DENOISE_NO_MOTION : PT_DENOISE_OFFSCREEN);
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
			const float footprint = 2.0f * previousDepth * fmaxf(tanX / params.width, tanY / params.height);
			float facing = 0.0f;
			for (int c = 0; c < 3; c++) facing += guide.normal[c] * previousParams.camForward[c];
			const float depthTolerance = fmaxf(0.01f, 2.0f * footprint / fmaxf(0.1f, fabsf(facing)));
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
						if (!oldGuide.historyValid) { rejection = PT_DENOISE_NO_MOTION; continue; }
						float planeDistance = 0.0f;
						for (int c = 0; c < 3; c++) planeDistance += (guide.position[c] - oldGuide.position[c]) * oldGuide.normal[c];
						if (oldGuide.depth <= 0.0f || fabsf(oldGuide.depth - previousDepth) > depthTolerance
							|| fabsf(planeDistance) > fmaxf(0.01f, footprint)) { rejection = PT_DENOISE_DEPTH; continue; }
						if (oldGuide.materialIndex != guide.materialIndex) { rejection = PT_DENOISE_MATERIAL; continue; }
						if (PT_DenoiseNormalDot(guide, oldGuide) < 0.9f) { rejection = PT_DENOISE_NORMAL; continue; }
						if (signal == 1) {
							float shadingDot = 0.0f, viewDot = 0.0f, viewLength = 0.0f, oldViewLength = 0.0f;
							for (int c = 0; c < 3; c++) {
								shadingDot += guide.shadingNormal[c] * oldGuide.shadingNormal[c];
								const float view = guide.position[c] - params.camPos[c];
								viewDot += view * delta[c];
								viewLength += view * view;
								oldViewLength += delta[c] * delta[c];
							}
							viewDot /= sqrtf(fmaxf(1e-12f, viewLength * oldViewLength));
							if (shadingDot < 0.98f - 0.18f * guide.roughness * guide.roughness
								|| viewDot < 1.0f - fmaxf(0.0001f, 0.05f * guide.roughness * guide.roughness)) {
								rejection = PT_DENOISE_NORMAL; continue;
							}
							if (fabsf(guide.roughness - oldGuide.roughness) > 0.1f) { rejection = PT_DENOISE_REFLECTANCE; continue; }
						}
						bool sameAlbedo = true;
						const float *reflectance = PT_DenoiseReflectance(guide, signal);
						const float *oldReflectance = PT_DenoiseReflectance(oldGuide, signal);
						for (int c = 0; c < 3; c++) {
							sameAlbedo &= fabsf(reflectance[c] - oldReflectance[c])
								<= 0.02f + 0.2f * fmaxf(reflectance[c], oldReflectance[c]);
						}
						if (!sameAlbedo) { rejection = PT_DENOISE_REFLECTANCE; continue; }
						if (previousMoments[neighbor * 4 + 2] < 1.0f) { rejection = PT_DENOISE_NO_SAMPLES; continue; }
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
			rejection = PT_DENOISE_REUSED;
			const float cap = signal == 0 ? 32.0f : 4.0f + 20.0f * guide.roughness * guide.roughness;
			historyLength = fminf(cap, historyMoments[2] + 1.0f);
			const float alpha = fmaxf(signal == 0 ? 0.08f : 0.35f - 0.25f * guide.roughness, 1.0f / historyLength);
			const float momentAlpha = fmaxf(0.1f, 1.0f / historyLength);
			mean += (historyMoments[0] - mean) * (1.0f - momentAlpha);
			square += (historyMoments[1] - square) * (1.0f - momentAlpha);
			for (int c = 0; c < 3; c++) {
				const float margin = 0.25f * (localMax[c] - localMin[c]);
				const float clipped = fminf(localMax[c] + margin, fmaxf(localMin[c] - margin, history[c]));
				current[c] += (clipped - current[c]) * (1.0f - alpha);
			}
		} else rejection = PT_DENOISE_LIGHTING;
	}
	float variance = fmaxf(0.0f, square - mean * mean);
	if (historyLength < 4.0f) variance = fmaxf(variance, localVariance) * (4.0f / historyLength);
	for (int c = 0; c < 3; c++) output[index + c] = current[c];
	output[index + 3] = variance;
	moments[index] = mean;
	moments[index + 1] = square;
	moments[index + 2] = historyLength;
	moments[index + 3] = (float)rejection;
}

__global__ void SVGFAtrousKernel(
	const float *input, const ptCudaDenoiseGuide_t *guides, float *output,
	int width, int height, int step, float strength, int signal
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
	if (signal == 1) step = min(step, max(1, (int)(1.0f + 7.0f * guide.roughness * guide.roughness)));

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
			float shadingDot = 0.0f;
			for (int c = 0; c < 3; c++) shadingDot += guide.shadingNormal[c] * sampleGuide.shadingNormal[c];
			shadingDot = fminf(1.0f, fmaxf(0.0f, shadingDot));
			const float normalWeight = dx == 0 && dy == 0 ? 1.0f : powf(normalDot, 32.0f)
				* powf(shadingDot, signal == 0 ? 4.0f : 128.0f - 96.0f * guide.roughness * guide.roughness);
			const float roughnessWeight = signal == 0 ? 1.0f : expf(-20.0f * fabsf(guide.roughness - sampleGuide.roughness));
			const float weight = (dx == 0 ? 2.0f : 1.0f) * (dy == 0 ? 2.0f : 1.0f)
				* normalWeight * roughnessWeight
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

__global__ void SVGFStoreDiffuseKernel(
	const float *illumination, const ptCudaDenoiseGuide_t *guides, float *output, int width, int height
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	const int pixel = y * width + x;
	for (int c = 0; c < 3; c++) output[pixel * 4 + c] = illumination[pixel * 4 + c] * guides[pixel].albedo[c];
	output[pixel * 4 + 3] = 0.0f;
}

__global__ void SVGFResolveKernel(
	const float *specular, const float *filteredDiffuse, const float *radiance, const float *diffuseRadiance,
	const float *specularRadiance, const float *accumulation, const ptCudaDenoiseGuide_t *guides,
	float *output, int width, int height, int debug
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	const int pixel = y * width + x;
	const int index = pixel * 4;
	for (int c = 0; c < 3; c++) {
		if (guides[pixel].depth > 0.0f) {
			const float filteredSpecular = specular[index + c] * guides[pixel].specularAlbedo[c];
			const float residual = fmaxf(0.0f, radiance[index + c] - diffuseRadiance[index + c] - specularRadiance[index + c]);
			output[index + c] = debug == 5 ? filteredDiffuse[index + c] : (debug == 6 ? filteredSpecular
				: filteredDiffuse[index + c] + filteredSpecular + residual);
		} else output[index + c] = debug >= 5 ? 0.0f : accumulation[index + c];
	}
	output[index + 3] = accumulation[index + 3];
}

__global__ void SVGFDenoiseDebugKernel(
	const ptCudaDenoiseGuide_t *guides, const float *moments, const float *history,
	unsigned char *output, int width, int height, int debug, int progressive
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	const int pixel = y * width + x, index = pixel * 4;
	float color[3] = {};
	if (debug == 1) {
		const int coverage = guides[pixel].coverage;
		color[0] = coverage >= PT_DENOISE_DYNAMIC ? 1.0f : 0.0f;
		color[1] = coverage == PT_DENOISE_STATIC || coverage == PT_DENOISE_DYNAMIC || coverage == PT_DENOISE_MIXED ? 1.0f : 0.0f;
		color[2] = coverage == PT_DENOISE_OVERLAY ? 1.0f : 0.0f;
		if (coverage == PT_DENOISE_MIXED) color[1] = 0.4f;
	} else if (guides[pixel].depth > 0.0f) {
		if (debug == 2) {
			const float t = fminf(1.0f, fmaxf(0.0f, (moments[index + 2] - 1.0f) / 31.0f));
			color[0] = 1.0f - t;
			color[1] = t;
		} else if (debug == 3) {
			const float variance = sqrtf(fmaxf(0.0f, progressive
				? moments[index + 1] - moments[index] * moments[index] : history[index + 3]));
			const float relative = variance / fmaxf(0.05f, progressive ? moments[index] : PT_DenoiseLuminance(history + index));
			color[0] = color[1] = color[2] = relative / (1.0f + relative);
		} else {
			const int reason = (int)moments[index + 3];
			switch (reason) {
				case PT_DENOISE_REUSED: color[1] = 1.0f; break;
				case PT_DENOISE_RESET: color[0] = color[2] = 1.0f; break;
				case PT_DENOISE_NO_MOTION: color[0] = color[1] = 1.0f; break;
				case PT_DENOISE_OFFSCREEN: color[2] = 1.0f; break;
				case PT_DENOISE_DEPTH: color[0] = 1.0f; break;
				case PT_DENOISE_NORMAL: color[1] = color[2] = 1.0f; break;
				case PT_DENOISE_MATERIAL: color[0] = 1.0f; color[1] = 0.5f; break;
				case PT_DENOISE_REFLECTANCE: color[0] = 0.5f; color[2] = 1.0f; break;
				case PT_DENOISE_LIGHTING: color[0] = color[1] = color[2] = 1.0f; break;
				case PT_DENOISE_NO_SAMPLES: color[0] = color[1] = color[2] = 0.5f; break;
			}
		}
	}
	for (int c = 0; c < 3; c++) output[index + c] = (unsigned char)(color[c] * 255.0f + 0.5f);
	output[index + 3] = 255;
}

#endif /* !__PT_KERNEL_DENOISING_CUH__ */
