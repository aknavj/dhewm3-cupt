#include "renderer_pt/cuda/pt_kernel_pathtracing.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void CheckCuda(cudaError_t result, const char* call, int line) {
	if (result != cudaSuccess) {
		std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, line, call, cudaGetErrorString(result));
		std::exit(1);
	}
}
#define CUDA_CHECK(call) CheckCuda((call), #call, __LINE__)

static void CheckNear(const char* label, double actual, double expected, double tolerance) {
	if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
		std::fprintf(stderr, "%s: got %.9f, expected %.9f (+/- %.9f)\n", label, actual, expected, tolerance);
		std::exit(1);
	}
}

static ptRenderParams_t PhysicalParams() {
	ptRenderParams_t p = {};
	p.physicalMetersPerUnit = 0.01f;
	p.physicalLightLumens = 4.0f * PT_PI * 100.0f;
	p.physicalSunLux = 50000.0f;
	p.physicalIor = 1.5f;
	p.physicalRoughnessScale = 1.0f;
	p.physicalFocusDistance = 300.0f;
	p.physicalApertureRadius = 0.625f;
	p.camForward[2] = 1.0f;
	p.camRight[0] = p.camUp[1] = 1.0f;
	return p;
}

__global__ void CheckLightModel(ptRenderParams_t p, float* results) {
	ptLight_t light = {};
	light.intensity = 1.0f;
	light.radius = 1.0f;
	light.boxFalloff = 1;
	light.position[2] = 100.0f;
	light.color[0] = light.color[1] = light.color[2] = 1.0f;
	light.projectedTextureIndex = light.falloffTextureIndex = -1;
	const float position[3] = { 0.0f, 0.0f, 0.0f };
	const float normal[3] = { 0.0f, 0.0f, 1.0f };
	float filter[3];
	results[0] = PT_LightFilter(light, position, 100, normal, NULL, filter, &p);
	results[1] = PT_LightFilter(light, position, 200, normal, NULL, filter, &p);
	p.physicalMetersPerUnit *= 2.0f;
	results[2] = PT_LightFilter(light, position, 100, normal, NULL, filter, &p);
	results[3] = LightImportance(light, position, normal, 100, &p) > 0.0f;
	ptRng_t rng;
	PT_RngInit(&rng, 0, 0);
	int picked = -1;
	float weight = 0.0f;
	results[4] = PT_SelectLights(&light, 1, position, normal, 1, 100, &rng, &picked, &weight, &p);
	results[5] = weight;
	light.type = PT_LIGHT_DIRECTIONAL;
	results[6] = PT_LightFilter(light, position, 100, normal, NULL, filter, &p);
	light.direction[2] = -1.0f;
	results[10] = LightImportance(light, position, normal, 100, &p);
	light.type = PT_LIGHT_POINT;
	light.coneAngle = PT_PI / 4.0f;
	light.coneFalloff = 2.0f;
	light.direction[2] = -1.0f;
	p.physicalMetersPerUnit = 0.01f;
	results[7] = PT_LightFilter(light, position, 100, normal, NULL, filter, &p);
	const float away[3] = { 0.0f, 0.0f, -1.0f };
	results[8] = PT_LightFilter(light, position, 100, away, NULL, filter, &p);
	results[9] = PT_LightFilter(light, position, 100, normal, NULL, filter);
}

__global__ void CheckLens(ptRenderParams_t p, float* results, int count) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) return;
	ptRng_t rng;
	PT_RngInit(&rng, index, 1);
	float origin[3] = {};
	float direction[3] = { 0.2f, -0.1f, 1.0f };
	normalize3(direction);
	PT_PhysicalLens(p, &rng, origin, direction);
	const float t = p.physicalFocusDistance / direction[2];
	results[index * 4] = origin[0] + t * direction[0];
	results[index * 4 + 1] = origin[1] + t * direction[1];
	results[index * 4 + 2] = origin[0] * origin[0] + origin[1] * origin[1];
	p.physicalApertureRadius = 0.0f;
	const unsigned int state = rng.state;
	PT_PhysicalLens(p, &rng, origin, direction);
	results[index * 4 + 3] = rng.state == state;
}

__global__ void IntegrateBsdf(float* results, int count, float roughness, float metallic, float viewCosine) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) return;
	ptRng_t rng;
	PT_RngInit(&rng, index, 9);
	const float normal[3] = { 0.0f, 0.0f, 1.0f };
	const float wo[3] = { sqrtf(1.0f - viewCosine * viewCosine), 0.0f, viewCosine };
	const float albedo[3] = { 1.0f, 0.5f, 0.1f };
	float wi[3], pdf, brdf[3], diffuse[3];
	PT_SamplePhysicalBsdf(normal, wo, albedo, metallic, roughness, 1.5f, &rng, wi, pdf);
	PT_EvaluatePhysicalMaterial(albedo, normal, wo, wi, metallic, roughness, 1.5f, brdf, diffuse);
	for (int c = 0; c < 3; c++) {
		results[index * 4 + c] = pdf > 0.0f ? brdf[c] * fmaxf(wi[2], 0.0f) / pdf : 0.0f;
	}
	float reciprocal[3];
	PT_EvaluatePhysicalMaterial(albedo, normal, wi, wo, metallic, roughness, 1.5f, reciprocal, NULL);
	results[index * 4 + 3] = fabsf(brdf[0] - reciprocal[0]) / fmaxf(1.0f, brdf[0]);
}

__global__ void CheckDirect(float* results) {
	ptVertex_t vertices[3] = {};
	vertices[0].position[0] = vertices[0].position[1] = -1000.0f;
	vertices[1].position[0] = 1000.0f;
	vertices[1].position[1] = -1000.0f;
	vertices[2].position[1] = 1000.0f;
	ptTriangle_t triangle = {};
	triangle.vertexIndices[1] = 1;
	triangle.vertexIndices[2] = 2;
	triangle.flags = PT_TRI_OPAQUE;
	ptBVHNode_t node = {};
	node.bounds[0] = node.bounds[1] = -1000.0f;
	node.bounds[3] = node.bounds[4] = 1000.0f;
	node.leftChild = node.escapeIndex = -1;
	node.primitiveCount = 1;
	const int index = 0;
	ptMaterial_t material = {};
	ptLight_t light = {};
	light.intensity = 1.0f;
	light.color[0] = light.color[1] = light.color[2] = 1.0f;
	light.position[2] = 100.0f;
	light.radius = 1000.0f;
	light.projectedTextureIndex = light.textureIndex = -1;
	ptRenderParams_t p = {};
	p.physicalMetersPerUnit = 0.01f;
	p.physicalLightLumens = 4.0f * PT_PI * 100.0f;
	p.physicalIor = 1.5f;
	const float position[3] = {}, normal[3] = { 0.0f, 0.0f, 1.0f };
	ptRng_t rng;
	PT_RngInit(&rng, 0, 0);
	float radiance[3], wi[3];
	SampleDirectLight(light, position, normal, 0, 100, vertices, &triangle, &node, &index, &material,
		NULL, &rng, radiance, wi, &p);
	results[0] = radiance[0];
	light.position[2] = 200.0f;
	SampleDirectLight(light, position, normal, 0, 100, vertices, &triangle, &node, &index, &material,
		NULL, &rng, radiance, wi, &p);
	results[1] = radiance[0];
}

static void CheckPathIntegration() {
	struct scene_t {
		ptVertex_t vertices[3];
		ptTriangle_t triangle;
		ptBVHNode_t node;
		ptMaterial_t material;
		ptLight_t light;
		int index;
	} scene = {};
	scene.vertices[0].position[0] = scene.vertices[0].position[1] = -1000;
	scene.vertices[1].position[0] = 1000;
	scene.vertices[1].position[1] = -1000;
	scene.vertices[2].position[1] = 1000;
	for (int i = 0; i < 3; i++) {
		scene.vertices[i].position[2] = 100;
		scene.vertices[i].normal[2] = -1;
	}
	scene.triangle.vertexIndices[1] = 1;
	scene.triangle.vertexIndices[2] = 2;
	scene.triangle.flags = PT_TRI_OPAQUE;
	scene.node.bounds[0] = scene.node.bounds[1] = -1000;
	scene.node.bounds[2] = scene.node.bounds[5] = 100;
	scene.node.bounds[3] = scene.node.bounds[4] = 1000;
	scene.node.leftChild = scene.node.escapeIndex = -1;
	scene.node.primitiveCount = 1;
	scene.material.albedoTexture = scene.material.emissionTexture = scene.material.specularTexture = scene.material.normalTexture = -1;
	scene.material.albedo[3] = 1;
	scene.material.roughness = 0.5f;
	scene.material.texTransform[0] = scene.material.texTransform[4] = 1;
	scene.light.intensity = 1;
	scene.light.textureIndex = scene.light.projectedTextureIndex = -1;
	for (int c = 0; c < 3; c++) {
		scene.material.albedo[c] = 1;
		scene.material.emission[c] = 1.0f / (1 << c);
		scene.light.color[c] = 1;
	}
	scene_t* device = NULL;
	float *accumulation = NULL, *diffuse = NULL, *specular = NULL;
	ptCudaDenoiseGuide_t* guide = NULL;
	CUDA_CHECK(cudaMalloc(&device, sizeof(scene)));
	CUDA_CHECK(cudaMalloc(&accumulation, 4 * sizeof(float)));
	CUDA_CHECK(cudaMemset(accumulation, 0, 4 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&diffuse, 4 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&specular, 4 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&guide, sizeof(*guide)));
	CUDA_CHECK(cudaMemcpy(device, &scene, sizeof(scene), cudaMemcpyHostToDevice));
	ptRenderParams_t p = PhysicalParams();
	p.width = p.height = 1;
	p.maxDepth = 1;
	p.samplesPerPixel = 4;
	p.maxLightSamples = 1;
	p.rayOffset = 0.001f;
	p.emissionBoost = 100;
	p.captureTargetFrames = 2;
	float result[4], d[4], s[4];
	ptCudaDenoiseGuide_t g;
	for (int frame = 0; frame < 2; frame++) {
		p.frameIndex = frame;
		p.sampleIndex = frame;
		PathTracingKernel<<<1, 1>>>(device->vertices, &device->triangle, &device->node, &device->index, &device->material,
			NULL, &device->light, 0, accumulation, p, NULL, guide, diffuse, specular);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaMemcpy(result, accumulation, sizeof(result), cudaMemcpyDeviceToHost));
		for (int c = 0; c < 3; c++) CheckNear("progressive physical emissive nits", result[c], 100.0f / (1 << c), 0.0001);
	}
	CUDA_CHECK(cudaMemcpy(&g, guide, sizeof(g), cudaMemcpyDeviceToHost));
	CheckNear("DOF rejects pinhole temporal history", g.historyValid, 0, 0);
	p.physicalApertureRadius = 0;
	p.frameIndex = 0;
	for (int c = 0; c < 3; c++) scene.material.emission[c] = 0;
	CUDA_CHECK(cudaMemcpy(device, &scene, sizeof(scene), cudaMemcpyHostToDevice));
	PathTracingKernel<<<1, 1>>>(device->vertices, &device->triangle, &device->node, &device->index, &device->material,
		NULL, &device->light, 1, accumulation, p, NULL, guide, diffuse, specular);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(result, accumulation, sizeof(result), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(d, diffuse, sizeof(d), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(s, specular, sizeof(s), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&g, guide, sizeof(g), cudaMemcpyDeviceToHost));
	const float expectedDiffuse = 100.0f * 0.96f * 0.96f / PT_PI;
	const float expectedSpecular = 100.0f * 0.04f / (4.0f * PT_PI * 0.0625f);
	CheckNear("physical path diffuse", d[0], expectedDiffuse, 0.001);
	CheckNear("physical path specular", s[0], expectedSpecular, 0.001);
	CheckNear("physical path signal conservation", result[0], d[0] + s[0], 0.001);
	CheckNear("pinhole permits temporal history", g.historyValid, 1, 0);
	CUDA_CHECK(cudaFree(device));
	CUDA_CHECK(cudaFree(accumulation));
	CUDA_CHECK(cudaFree(diffuse));
	CUDA_CHECK(cudaFree(specular));
	CUDA_CHECK(cudaFree(guide));
}

int main() {
	CheckNear("EV100 sunny 16", PT_PhysicalEV100(16, 1.0f / 125, 100), std::log2(32000.0), 0.00001);
	const float exposure = PT_PhysicalExposure(2.8f, 1.0f / 30, 800, 0);
	CheckNear("ISO doubles exposure", PT_PhysicalExposure(2.8f, 1.0f / 30, 1600, 0) / exposure, 2, 0.00001);
	CheckNear("shutter doubles exposure", PT_PhysicalExposure(2.8f, 1.0f / 15, 800, 0) / exposure, 2, 0.00001);
	CheckNear("stop doubles exposure", PT_PhysicalExposure(2.8f, 1.0f / 30, 800, 1) / exposure, 2, 0.00001);
	CheckNear("aperture halves exposure", PT_PhysicalExposure(2.8f * std::sqrt(2.0f), 1.0f / 30, 800, 0) / exposure, 0.5, 0.00001);
	CheckNear("50mm full-frame FOV", PT_PhysicalFov(36, 50), 39.59775, 0.0001);
	CheckNear("physical aperture radius", PT_PhysicalApertureRadius(50, 4, 0.01f), 0.625, 0.00001);
	ptRenderParams_t p = PhysicalParams(), changed = p;
	changed.exposure = 2;
	CheckNear("exposure preserves linear history", PT_PhysicalTraceChanged(p, changed), 0, 0);
	changed.physicalApertureRadius *= 2;
	CheckNear("aperture invalidates history", PT_PhysicalTraceChanged(p, changed), 1, 0);
	changed = p;
	changed.physicalFocusDistance *= 2;
	CheckNear("focus invalidates history", PT_PhysicalTraceChanged(p, changed), 1, 0);
	int devices = 0;
	cudaError_t availability = cudaGetDeviceCount(&devices);
	if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver || (availability == cudaSuccess && devices == 0)) {
		std::puts("SKIP: CUDA device unavailable");
		return 77;
	}
	CUDA_CHECK(availability);
	CheckPathIntegration();
	const int count = 262144;
	float* device = NULL;
	CUDA_CHECK(cudaMalloc(&device, count * 4 * sizeof(float)));
	std::vector<float> results(count * 4);
	CheckLightModel<<<1, 1>>>(p, device);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(results.data(), device, 11 * sizeof(float), cudaMemcpyDeviceToHost));
	CheckNear("100 lux at one meter", results[0], 100, 0.0001);
	CheckNear("inverse-square distance", results[1], 25, 0.0001);
	CheckNear("world scale affects illuminance", results[2], 25, 0.0001);
	CheckNear("importance outside legacy volume", results[3], 1, 0);
	CheckNear("light selection outside legacy volume", results[4], 1, 0);
	CheckNear("selection PDF", results[5], 1, 0.00001);
	CheckNear("directional lux", results[6], 50000, 0.01);
	CheckNear("directional photometric importance", results[10], 51000, 0.01);
	const float solidAngle = 2 * PT_PI * (1 - std::cos(PT_PI / 4)) / 3;
	CheckNear("normalized spot flux", results[7], p.physicalLightLumens / solidAngle, 0.01);
	CheckNear("spot rejects back hemisphere", results[8], 0, 0);
	CheckNear("legacy filter still clips volume", results[9], 0, 0);
	CheckDirect<<<1, 1>>>(device);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(results.data(), device, 2 * sizeof(float), cudaMemcpyDeviceToHost));
	CheckNear("unclamped physical direct light", results[0], 100, 0.001);
	CheckNear("direct inverse-square distance", results[1], 25, 0.001);
	CheckLens<<<count / 256, 256>>>(p, device, count);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(results.data(), device, results.size() * sizeof(float), cudaMemcpyDeviceToHost));
	double meanRadiusSquared = 0;
	for (int i = 0; i < count; i++) {
		CheckNear("lens focus X", results[i * 4], 60, 0.0001);
		CheckNear("lens focus Y", results[i * 4 + 1], -30, 0.0001);
		CheckNear("lens support", std::fmax(0.0f, results[i * 4 + 2] - p.physicalApertureRadius * p.physicalApertureRadius), 0, 0.00001);
		CheckNear("pinhole preserves RNG", results[i * 4 + 3], 1, 0);
		meanRadiusSquared += results[i * 4 + 2] / count;
	}
	CheckNear("uniform aperture area", meanRadiusSquared, p.physicalApertureRadius * p.physicalApertureRadius * 0.5, 0.001);
	for (float roughness : { 0.04f, 0.2f, 0.6f, 1.0f }) {
		for (float metallic : { 0.0f, 1.0f }) {
			for (float cosine : { 1.0f, 0.2f }) {
				IntegrateBsdf<<<count / 256, 256>>>(device, count, roughness, metallic, cosine);
				CUDA_CHECK(cudaGetLastError());
				CUDA_CHECK(cudaMemcpy(results.data(), device, results.size() * sizeof(float), cudaMemcpyDeviceToHost));
				double energy[3] = {};
				for (int i = 0; i < count; i++) {
					for (int c = 0; c < 3; c++) {
						const float value = results[i * 4 + c];
						if (!std::isfinite(value) || value < 0) {
							std::fprintf(stderr, "Invalid physical BSDF sample\n");
							return 1;
						}
						energy[c] += value / count;
					}
					CheckNear("BSDF reciprocity", results[i * 4 + 3], 0, 0.0001);
				}
				std::printf("roughness %.2f metal %.0f view %.1f: reflected energy %.5f %.5f %.5f\n",
					roughness, metallic, cosine, energy[0], energy[1], energy[2]);
				for (double e : energy) {
					if (e < 0.001 || e > 1.02) {
						std::fprintf(stderr, "BSDF failed energy bounds: %.8f\n", e);
						return 1;
					}
				}
			}
		}
	}
	CUDA_CHECK(cudaFree(device));
	std::puts("Physical lighting, camera, sampling and energy-conservation tests passed");
	return 0;
}
