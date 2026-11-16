#include "renderer_pt/cuda/pt_cuda_device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void CheckCuda(cudaError_t error, const char* call, int line) {
	if (error != cudaSuccess) {
		std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, line, call, cudaGetErrorString(error));
		std::exit(1);
	}
}

#define CUDA_CHECK(call) CheckCuda((call), #call, __LINE__)

static void CheckNear(const char* label, float actual, float expected, float tolerance = 1e-5f) {
	if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
		std::fprintf(stderr, "%s: got %.9f, expected %.9f (+/- %.9f)\n", label, actual, expected, tolerance);
		std::exit(1);
	}
}

template<class T>
static T* Allocate(size_t count) {
	T* result = NULL;
	CUDA_CHECK(cudaMalloc(&result, count * sizeof(T)));
	CUDA_CHECK(cudaMemset(result, 0xff, count * sizeof(T)));
	return result;
}

struct fixture_t {
	ptRenderParams_t params, previousParams;
	ptCudaDenoiseBuffers_t buffers;
	std::vector<float> radiance, accumulation, lastMoments;
	std::vector<ptCudaDenoiseGuide_t> guides;
	float *d_filtered, *d_accumulation;
	cudaStream_t stream;
	int slot;
	bool haveHistory;

	fixture_t(int width, int height) : params(), previousParams(), buffers(),
		radiance(width * height * 4, 1.0f), lastMoments(width * height * 4), guides(width * height),
		slot(0), haveHistory(false) {
		params.width = width;
		params.height = height;
		params.camForward[2] = params.camRight[0] = params.camUp[1] = 1.0f;
		params.fovX = params.fovY = 90.0f;
		params.denoiseStrength = 0.5f;
		for (int i = 0; i < width * height; i++) {
			radiance[i * 4 + 3] = 0.75f;
			guides[i].normal[2] = 1.0f;
			guides[i].depth = 10.0f;
			guides[i].albedo[0] = guides[i].albedo[1] = guides[i].albedo[2] = 1.0f;
			guides[i].historyValid = 1;
		}
		UpdatePositions();
		CUDA_CHECK(cudaStreamCreate(&stream));
		buffers.radiance = Allocate<float>(radiance.size());
		buffers.scratch = Allocate<float>(radiance.size());
		d_filtered = Allocate<float>(radiance.size());
		d_accumulation = Allocate<float>(radiance.size());
		for (int i = 0; i < 2; i++) {
			buffers.guides[i] = Allocate<ptCudaDenoiseGuide_t>(guides.size());
			buffers.history[i] = Allocate<float>(radiance.size());
			buffers.moments[i] = Allocate<float>(radiance.size());
		}
	}

	~fixture_t() {
		CUDA_CHECK(cudaFree(buffers.radiance));
		CUDA_CHECK(cudaFree(buffers.scratch));
		CUDA_CHECK(cudaFree(d_filtered));
		CUDA_CHECK(cudaFree(d_accumulation));
		for (int i = 0; i < 2; i++) {
			CUDA_CHECK(cudaFree(buffers.guides[i]));
			CUDA_CHECK(cudaFree(buffers.history[i]));
			CUDA_CHECK(cudaFree(buffers.moments[i]));
		}
		CUDA_CHECK(cudaStreamDestroy(stream));
	}

	void UpdatePositions() {
		for (int y = 0; y < params.height; y++) {
			for (int x = 0; x < params.width; x++) {
				ptCudaDenoiseGuide_t& guide = guides[y * params.width + x];
				guide.position[0] = params.camPos[0] + (2.0f * (x + 0.5f) / params.width - 1.0f) * guide.depth;
				guide.position[1] = params.camPos[1] + (2.0f * (y + 0.5f) / params.height - 1.0f) * guide.depth;
				guide.position[2] = params.camPos[2] + guide.depth;
			}
		}
	}

	std::vector<float> Filter(bool reuseHistory = true) {
		const std::vector<float>& raw = accumulation.empty() ? radiance : accumulation;
		CUDA_CHECK(cudaMemcpyAsync(buffers.radiance, radiance.data(), radiance.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_accumulation, raw.data(), raw.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(buffers.guides[slot], guides.data(), guides.size() * sizeof(ptCudaDenoiseGuide_t), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(LaunchDenoisingKernel(&buffers, d_accumulation, d_filtered, &params, &previousParams,
			slot, haveHistory && reuseHistory ? 1 : 0, stream));
		std::vector<float> result(radiance.size()), original(radiance.size()), originalRaw(raw.size());
		CUDA_CHECK(cudaMemcpyAsync(result.data(), d_filtered, result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(original.data(), buffers.radiance, original.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(originalRaw.data(), d_accumulation, originalRaw.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(lastMoments.data(), buffers.moments[slot], lastMoments.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaStreamSynchronize(stream));
		for (size_t i = 0; i < radiance.size(); i++) {
			CheckNear("fresh radiance unchanged", original[i], radiance[i], 0.0f);
			CheckNear("progressive accumulation unchanged", originalRaw[i], raw[i], 0.0f);
			if (!std::isfinite(result[i])) {
				std::fprintf(stderr, "Nonfinite filtered output at %zu\n", i);
				std::exit(1);
			}
			if (i % 4 == 3) CheckNear("alpha preserved", result[i], radiance[i], 0.0f);
		}
		slot ^= 1;
		haveHistory = true;
		previousParams = params;
		return result;
	}
};

static void CheckConstant(int width, int height) {
	fixture_t f(width, height);
	for (int i = 0; i < width * height; i++) {
		f.radiance[i * 4] = 8.0f;
		f.radiance[i * 4 + 1] = 0.125f;
		f.radiance[i * 4 + 2] = 0.0f;
	}
	for (int frame = 0; frame < 3; frame++) {
		const std::vector<float> result = f.Filter();
		for (size_t i = 0; i < result.size(); i++) CheckNear("constant HDR including borders", result[i], f.radiance[i]);
	}
}

static double SquaredError(const std::vector<float>& color) {
	double error = 0.0;
	for (size_t i = 0; i < color.size(); i += 4) error += std::pow(color[i] - 1.0f, 2);
	return error;
}

static void SetNoise(fixture_t& f, int frame) {
	for (int y = 0; y < f.params.height; y++) {
		for (int x = 0; x < f.params.width; x++) {
			for (int c = 0; c < 3; c++) {
				f.radiance[(y * f.params.width + x) * 4 + c] = (x + y + frame) % 2 ? 0.75f : 1.25f;
			}
		}
	}
}

static void CheckSmoothing() {
	fixture_t f(19, 13);
	SetNoise(f, 0);
	const std::vector<float> result = f.Filter();
	const double before = SquaredError(f.radiance), after = SquaredError(result);
	std::printf("Flat-surface squared error: %.6f -> %.6f\n", before, after);
	if (after >= before * 0.25) {
		std::fprintf(stderr, "SVGF must reduce flat-surface squared error by at least 75%%\n");
		std::exit(1);
	}
	f.params.denoiseStrength = 0.05f;
	if (SquaredError(f.Filter(false)) <= after) {
		std::fprintf(stderr, "Lower strength must preserve more illumination contrast\n");
		std::exit(1);
	}
	f.params.denoiseStrength = 0.5f;
	for (int frame = 1; frame < 16; frame++) {
		SetNoise(f, frame);
		f.Filter();
	}
	for (size_t i = 0; i < f.guides.size(); i++) {
		CheckNear("temporal history reused", f.lastMoments[i * 4 + 2], 16.0f);
	}

	unsigned char* d_raw = Allocate<unsigned char>(f.radiance.size());
	unsigned char* d_denoised = Allocate<unsigned char>(f.radiance.size());
	LaunchToneMappingKernel(f.buffers.radiance, d_raw, f.params.width, f.params.height, 1.0f, 2.2f, 1, f.stream);
	CUDA_CHECK(cudaGetLastError());
	LaunchToneMappingKernel(f.d_filtered, d_denoised, f.params.width, f.params.height, 1.0f, 2.2f, 1, f.stream);
	CUDA_CHECK(cudaGetLastError());
	std::vector<unsigned char> raw(f.radiance.size()), denoised(f.radiance.size());
	CUDA_CHECK(cudaMemcpyAsync(raw.data(), d_raw, raw.size(), cudaMemcpyDeviceToHost, f.stream));
	CUDA_CHECK(cudaMemcpyAsync(denoised.data(), d_denoised, denoised.size(), cudaMemcpyDeviceToHost, f.stream));
	CUDA_CHECK(cudaStreamSynchronize(f.stream));
	if (raw == denoised) {
		std::fprintf(stderr, "SVGF must affect tone-mapped output\n");
		std::exit(1);
	}
	CUDA_CHECK(cudaFree(d_raw));
	CUDA_CHECK(cudaFree(d_denoised));
}

static void CheckEdgesAndTextures() {
	for (int mode = 0; mode < 3; mode++) {
		fixture_t f(17, 9);
		for (int y = 0; y < f.params.height; y++) {
			for (int x = f.params.width / 2; x < f.params.width; x++) {
				const int pixel = y * f.params.width + x;
				for (int c = 0; c < 3; c++) f.radiance[pixel * 4 + c] = 1.5f;
				if (mode == 0) f.guides[pixel].depth = 20.0f;
				if (mode == 1) {
					f.guides[pixel].normal[0] = 1.0f;
					f.guides[pixel].normal[2] = 0.0f;
				}
				if (mode == 2) f.guides[pixel].depth = 0.0f;
			}
		}
		f.params.denoiseStrength = 4.0f;
		const std::vector<float> result = f.Filter();
		for (size_t i = 0; i < result.size(); i++) CheckNear("depth/normal/bypass edges", result[i], f.radiance[i]);
	}
	fixture_t f(19, 13);
	for (size_t i = 0; i < f.guides.size(); i++) {
		for (int c = 0; c < 3; c++) {
			f.guides[i].albedo[c] = (i + c) % 2 ? 0.04f : 1.0f;
			f.radiance[i * 4 + c] = f.guides[i].albedo[c] * (c + 1.0f);
		}
	}
	for (int frame = 0; frame < 3; frame++) {
		const std::vector<float> result = f.Filter();
		for (size_t i = 0; i < result.size(); i++) CheckNear("albedo detail restored after filtering", result[i], f.radiance[i]);
	}
}

static void CheckReprojection() {
	fixture_t f(20, 12);
	SetNoise(f, 0);
	f.Filter();
	f.params.camPos[0] = 1.0f; // exactly one pixel at depth ten and 90-degree FOV
	f.UpdatePositions();
	SetNoise(f, 1);
	f.Filter();
	for (int y = 0; y < f.params.height; y++) {
		for (int x = 0; x < f.params.width; x++) {
			CheckNear("camera translation reprojection and disocclusion",
				f.lastMoments[(y * f.params.width + x) * 4 + 2], x + 1 < f.params.width ? 2.0f : 1.0f);
			const int index = (y * f.params.width + x) * 4;
			CheckNear("reprojected moment comes from the translated pixel", f.lastMoments[index], f.radiance[index]);
			CheckNear("reprojected second moment", f.lastMoments[index + 1], f.radiance[index] * f.radiance[index]);
		}
	}
}

static void CheckTemporalContribution() {
	fixture_t temporal(19, 13), spatial(19, 13);
	SetNoise(temporal, 0);
	SetNoise(spatial, 0);
	temporal.Filter();
	spatial.Filter();
	SetNoise(temporal, 1);
	SetNoise(spatial, 1);
	const double temporalError = SquaredError(temporal.Filter());
	const double spatialError = SquaredError(spatial.Filter(false));
	std::printf("Second-frame squared error: temporal %.6f, spatial-only %.6f\n", temporalError, spatialError);
	if (temporalError >= spatialError) {
		std::fprintf(stderr, "History must improve reconstruction, not just increment a counter\n");
		std::exit(1);
	}
}

static void CheckFreshSignalAndFallback() {
	fixture_t f(9, 7);
	f.accumulation = f.radiance;
	for (size_t i = 0; i < f.guides.size(); i++) {
		if (i % 2) f.guides[i].depth = 0.0f;
		for (int c = 0; c < 3; c++) f.accumulation[i * 4 + c] = 20.0f;
	}
	const std::vector<float> result = f.Filter();
	for (size_t i = 0; i < f.guides.size(); i++) {
		for (int c = 0; c < 3; c++) {
			CheckNear("fresh SVGF signal, raw progressive fallback", result[i * 4 + c], i % 2 ? 20.0f : 1.0f);
		}
		if (i % 2) CheckNear("bypass stores no history", f.lastMoments[i * 4 + 2], 0.0f);
	}
	fixture_t degenerate(9, 7);
	for (size_t i = 0; i < degenerate.guides.size(); i++) degenerate.guides[i].normal[2] = 0.0f;
	const std::vector<float> unfiltered = degenerate.Filter();
	for (size_t i = 0; i < unfiltered.size(); i++) {
		CheckNear("degenerate normals retain a finite center sample", unfiltered[i], degenerate.radiance[i]);
	}
}

static void CheckHistoryRejection() {
	for (int mode = 0; mode < 8; mode++) {
		fixture_t f(17, 9);
		if (mode == 7) {
			for (size_t i = 0; i < f.guides.size(); i++) f.guides[i].historyValid = 0;
		}
		f.Filter();
		for (size_t i = 0; i < f.guides.size(); i++) {
			if (mode == 0) f.guides[i].depth = 20.0f;
			if (mode == 1) {
				f.guides[i].normal[0] = 1.0f;
				f.guides[i].normal[2] = 0.0f;
			}
			if (mode == 2) f.guides[i].materialIndex = 1;
			if (mode == 3) {
				for (int c = 0; c < 3; c++) f.guides[i].albedo[c] = f.radiance[i * 4 + c] = 0.5f;
			}
			if (mode == 4) f.guides[i].historyValid = 0;
			if (mode == 5) f.guides[i].position[0] += 1000.0f;
			if (mode == 7) f.guides[i].historyValid = 1;
		}
		if (mode == 0) f.UpdatePositions();
		f.Filter(mode != 6);
		for (size_t i = 0; i < f.guides.size(); i++) CheckNear("incompatible history rejected", f.lastMoments[i * 4 + 2], 1.0f);
	}
	fixture_t f(19, 13);
	for (int frame = 0; frame < 40; frame++) f.Filter();
	for (size_t i = 0; i < f.guides.size(); i++) CheckNear("history bounded", f.lastMoments[i * 4 + 2], 32.0f);
	for (size_t i = 0; i < f.guides.size(); i++) {
		for (int c = 0; c < 3; c++) f.radiance[i * 4 + c] = 2.0f;
	}
	const std::vector<float> changed = f.Filter();
	for (size_t i = 0; i < f.guides.size(); i++) {
		CheckNear("lighting change rejects stale history", f.lastMoments[i * 4 + 2], 1.0f);
		CheckNear("lighting change has no stale trail", changed[i * 4], 2.0f);
	}
}

int main() {
	int deviceCount = 0;
	const cudaError_t error = cudaGetDeviceCount(&deviceCount);
	if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver
		|| (error == cudaSuccess && deviceCount == 0)) {
		std::fprintf(stderr, "Skipping CUDA denoising test: no CUDA device or driver\n");
		return 77;
	}
	CUDA_CHECK(error);
	CheckConstant(1, 1);
	CheckConstant(1, 19);
	CheckConstant(19, 1);
	CheckConstant(19, 13);
	CheckSmoothing();
	CheckEdgesAndTextures();
	CheckReprojection();
	CheckTemporalContribution();
	CheckFreshSignalAndFallback();
	CheckHistoryRejection();
	std::puts("CUDA SVGF regression checks passed");
	return 0;
}
