#include "renderer_pt/cuda/pt_kernel_pathtracing.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void CheckCuda(cudaError_t error, const char* call, int line) {
	if (error != cudaSuccess) {
		std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, line, call, cudaGetErrorString(error));
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

template<class T>
static T* Allocate(size_t count) {
	T* result = NULL;
	CUDA_CHECK(cudaMalloc(&result, count * sizeof(T)));
	return result;
}

struct fixture_t {
	ptVertex_t vertices[6];
	ptTriangle_t triangles[2];
	ptMaterial_t materials[2];
	ptLight_t lights[2];
	ptBVHNode_t node;
	int indices[2];
	cudaTexture_t textures[2];
	unsigned char texels[2][8];

	ptVertex_t* d_vertices;
	ptTriangle_t* d_triangles;
	ptMaterial_t* d_materials;
	ptLight_t* d_lights;
	ptBVHNode_t* d_node;
	int* d_indices;
	cudaTexture_t* d_textures;
	unsigned char* d_texels[2];

	fixture_t() {
		std::memset(vertices, 0, sizeof(vertices));
		std::memset(triangles, 0, sizeof(triangles));
		std::memset(materials, 0, sizeof(materials));
		std::memset(lights, 0, sizeof(lights));
		std::memset(&node, 0, sizeof(node));

		const float positions[6][3] = {
			{ -1.0f, -1.0f, 0.0f }, { 1.0f, -1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f },
			{ -100.0f, -100.0f, 0.5f }, { 100.0f, -100.0f, 0.5f }, { 0.0f, 100.0f, 0.5f }
		};
		for (int i = 0; i < 6; i++) {
			std::memcpy(vertices[i].position, positions[i], sizeof(positions[i]));
			vertices[i].normal[2] = 1.0f;
			for (int c = 0; c < 4; c++) vertices[i].color[c] = 1.0f;
		}
		for (int i = 0; i < 2; i++) {
			for (int c = 0; c < 3; c++) triangles[i].vertexIndices[c] = i * 3 + c;
			triangles[i].materialIndex = i;
			triangles[i].flags = PT_TRI_OPAQUE;
			indices[i] = i;
			for (int c = 0; c < 4; c++) materials[i].albedo[c] = 1.0f;
			materials[i].albedoTexture = materials[i].normalTexture = -1;
			materials[i].specularTexture = materials[i].emissionTexture = -1;
			materials[i].texTransform[0] = materials[i].texTransform[4] = 1.0f;
		}
		triangles[0].flags |= PT_TRI_AREA_LIGHT;
		for (int c = 0; c < 3; c++) materials[0].emission[c] = 1.0f;

		lights[0].type = PT_LIGHT_AREA;
		lights[0].position[1] = -1.0f / 3.0f;
		lights[0].direction[2] = 1.0f;
		lights[0].area[0] = 2.0f;
		lights[0].areaBoundRadius = 4.0f / 3.0f;
		lights[0].areaTriangleIndex = 0;
		for (int c = 0; c < 3; c++) lights[0].color[c] = 1.0f;
		lights[1] = lights[0];
		for (int c = 0; c < 3; c++) lights[1].color[c] = 0.2f;

		node.bounds[0] = node.bounds[1] = -1.0f;
		node.bounds[3] = node.bounds[4] = 1.0f;
		node.leftChild = -1;
		node.escapeIndex = -1;
		node.primitiveCount = 1;

		const unsigned char albedo[8] = { 255, 255, 255, 64, 255, 255, 255, 255 };
		const unsigned char glow[8] = { 128, 128, 128, 255, 255, 255, 255, 255 };
		std::memcpy(texels[0], albedo, sizeof(albedo));
		std::memcpy(texels[1], glow, sizeof(glow));

		d_vertices = Allocate<ptVertex_t>(6);
		d_triangles = Allocate<ptTriangle_t>(2);
		d_materials = Allocate<ptMaterial_t>(2);
		d_lights = Allocate<ptLight_t>(2);
		d_node = Allocate<ptBVHNode_t>(1);
		d_indices = Allocate<int>(2);
		d_textures = Allocate<cudaTexture_t>(2);
		for (int i = 0; i < 2; i++) {
			d_texels[i] = Allocate<unsigned char>(8);
			textures[i].width = 2;
			textures[i].height = 1;
			textures[i].data = d_texels[i];
		}
	}

	~fixture_t() {
		CUDA_CHECK(cudaFree(d_vertices));
		CUDA_CHECK(cudaFree(d_triangles));
		CUDA_CHECK(cudaFree(d_materials));
		CUDA_CHECK(cudaFree(d_lights));
		CUDA_CHECK(cudaFree(d_node));
		CUDA_CHECK(cudaFree(d_indices));
		CUDA_CHECK(cudaFree(d_textures));
		for (int i = 0; i < 2; i++) CUDA_CHECK(cudaFree(d_texels[i]));
	}

	void Upload() {
		CUDA_CHECK(cudaMemcpy(d_vertices, vertices, sizeof(vertices), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_triangles, triangles, sizeof(triangles), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_materials, materials, sizeof(materials), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_lights, lights, sizeof(lights), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_node, &node, sizeof(node), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_indices, indices, sizeof(indices), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_textures, textures, sizeof(textures), cudaMemcpyHostToDevice));
		for (int i = 0; i < 2; i++) {
			CUDA_CHECK(cudaMemcpy(d_texels[i], texels[i], sizeof(texels[i]), cudaMemcpyHostToDevice));
		}
	}
};

__global__ void AreaSamples(
	const ptVertex_t* vertices, const ptTriangle_t* triangles, const ptMaterial_t* materials,
	const ptLight_t* lights, const ptBVHNode_t* node, const int* indices, const cudaTexture_t* textures,
	float height, float boost, bool selectTwo, float* result, int count
) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) return;
	ptRng_t rng;
	PT_RngInit(&rng, index, 7);
	const float position[3] = { 0.0f, 0.0f, height };
	const float normal[3] = { 0.0f, 0.0f, height > 0.0f ? -1.0f : 1.0f };
	float invPdf = 1.0f;
	int pick = 0;
	if (selectTwo && PT_SelectLights(lights, 2, position, normal, 1, boost, &rng, &pick, &invPdf) == 0) {
		result[index] = 0.0f;
		return;
	}
	float radiance[3], wi[3];
	SampleDirectLight(lights[pick], position, normal, 0.0f, boost, vertices, triangles,
		node, indices, materials, textures, &rng, radiance, wi);
	result[index] = radiance[0] * invPdf / PT_PI;
}

__global__ void HelperChecks(
	const ptVertex_t* vertices, const ptTriangle_t* triangles, const ptMaterial_t* materials,
	const ptLight_t* lights, const ptBVHNode_t* node, const int* indices, const cudaTexture_t* textures,
	float* result
) {
	const float position[3] = { 0.0f, 0.0f, 1.0f };
	const float farPosition[3] = { 0.0f, 0.0f, 1000.0f };
	const float normal[3] = { 0.0f, 0.0f, -1.0f };
	const float away[3] = { 0.0f, 0.0f, 1.0f };
	result[0] = LightImportance(lights[0], farPosition, normal, 1.0f);
	result[1] = LightImportance(lights[0], farPosition, away, 1.0f);
	result[2] = LightImportance(lights[0], position, normal, 0.0f);

	ptRng_t rng;
	PT_RngInit(&rng, 0, 0);
	float indirect[3];
	SampleIndirectLight(position, normal, normal, 0.0f, 2.0f, false, vertices, triangles,
		node, indices, materials, textures, NULL, 0, &rng, indirect);
	result[3] = indirect[0];
	SampleIndirectLight(position, normal, normal, 0.0f, 2.0f, true, vertices, triangles,
		node, indices, materials, textures, NULL, 0, &rng, indirect);
	result[4] = indirect[0];
	float attenuation[3] = { 1.0f, 1.0f, 1.0f };
	result[5] = PT_ShadowTransmit(normal, vertices, triangles[0], 0.25f, 0.25f,
		materials, textures, attenuation) ? 1.0f : 0.0f;
}

__global__ void ProjectionChecks(const cudaTexture_t* textures, float* result) {
	ptLight_t light = {};
	light.type = PT_LIGHT_POINT;
	light.boxFalloff = 1;
	light.lightProject[0][0] = 1.0f;
	light.lightProject[1][1] = 1.0f;
	light.lightProject[2][3] = 1.0f;
	light.lightProject[3][3] = 0.5f;
	light.projectedTextureIndex = 1;
	light.falloffTextureIndex = -1;
	light.projectedTextureTransform[0] = light.projectedTextureTransform[4] = 1.0f;
	light.color[0] = light.color[1] = light.color[2] = 1.0f;
	float position[3] = { 0.25f, 0.5f, 0.0f };
	const float direction[3] = { 0.0f, 0.0f, 1.0f };
	float filter[3];

	PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	result[0] = filter[0];
	light.projectedTextureTransform[2] = 0.5f;
	PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	result[1] = filter[0];
	light.projectedTextureTransform[2] = 1.0f;
	PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	result[2] = filter[0];
	light.projectedTextureAddress = PT_LIGHT_TEXTURE_CLAMP;
	PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	result[3] = filter[0];
	light.projectedTextureAddress = PT_LIGHT_TEXTURE_BORDER;
	PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	result[4] = filter[0];
	light.projectedTextureTransform[2] = 0.75f;
	PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	result[5] = filter[0];
	position[0] = 1.25f;
	light.projectedTextureTransform[2] = -1.0f;
	result[6] = PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	position[0] = 0.25f;
	light.projectedTextureIndex = -1;
	light.falloffTextureIndex = 1;
	light.lightProject[3][3] = 0.25f;
	result[7] = PT_LightFilter(light, position, 1.0f, direction, textures, filter);
	light.falloffTextureIndex = -1;
	result[8] = PT_LightFilter(light, position, 1.0f, direction, textures, filter);
}

static std::vector<float> Sample(fixture_t& fixture, float height, float boost, bool selectTwo = false) {
	const int count = 131072;
	fixture.Upload();
	float* d_result = Allocate<float>(count);
	AreaSamples<<<(count + 255) / 256, 256>>>(fixture.d_vertices, fixture.d_triangles, fixture.d_materials,
		fixture.d_lights, fixture.d_node, fixture.d_indices, fixture.d_textures, height, boost, selectTwo, d_result, count);
	CUDA_CHECK(cudaGetLastError());
	std::vector<float> result(count);
	CUDA_CHECK(cudaMemcpy(result.data(), d_result, count * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_result));
	return result;
}

static double Mean(const std::vector<float>& values) {
	double total = 0.0;
	for (size_t i = 0; i < values.size(); i++) total += values[i];
	return total / values.size();
}

static double ReferenceIntegral(double height, bool vertexMask = false, bool textured = false) {
	const int n = 700;
	double total = 0.0;
	for (int i = 0; i < n; i++) {
		const double root = std::sqrt((i + 0.5) / n);
		for (int j = 0; j < n; j++) {
			const double r = (j + 0.5) / n;
			const double b0 = 1.0 - root, b1 = root * (1.0 - r), b2 = root * r;
			if (vertexMask && b1 + b2 < 0.5) continue;
			const double x = -b0 + b1, y = -b0 - b1 + b2;
			const double dist2 = x * x + y * y + height * height;
			const double emission = textured ? std::pow((128.0 / 255.0) * (1.0 - b1) + b1, 2.2) : 1.0;
			total += emission * 2.0 * height * height / (3.141592653589793 * dist2 * dist2);
		}
	}
	return total / (n * n);
}

static std::vector<float> Helpers(fixture_t& fixture) {
	fixture.Upload();
	float* d_result = Allocate<float>(6);
	HelperChecks<<<1, 1>>>(fixture.d_vertices, fixture.d_triangles, fixture.d_materials,
		fixture.d_lights, fixture.d_node, fixture.d_indices, fixture.d_textures, d_result);
	CUDA_CHECK(cudaGetLastError());
	std::vector<float> result(6);
	CUDA_CHECK(cudaMemcpy(result.data(), d_result, 6 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_result));
	return result;
}

static double TracePath(fixture_t& fixture, int maxDepth, int maxLightSamples, float indirectProb, float emissionBoost) {
	fixture.Upload();
	ptRenderParams_t params = {};
	params.camPos[2] = 1.0f;
	params.camForward[2] = -1.0f;
	params.camRight[0] = 1.0f;
	params.camUp[1] = 1.0f;
	params.width = params.height = 256;
	params.samplesPerPixel = 1;
	params.maxDepth = maxDepth;
	params.maxLightSamples = maxLightSamples;
	params.indirectProb = indirectProb;
	params.rayOffset = 0.0001f;
	params.emissionBoost = emissionBoost;

	const int pixels = params.width * params.height;
	float* d_framebuffer = Allocate<float>(pixels * 4);
	CUDA_CHECK(cudaMemset(d_framebuffer, 0, pixels * 4 * sizeof(float)));
	const dim3 block(16, 16);
	const dim3 grid(params.width / block.x, params.height / block.y);
	PathTracingKernel<<<grid, block>>>(fixture.d_vertices, fixture.d_triangles, fixture.d_node, fixture.d_indices,
		fixture.d_materials, fixture.d_textures, fixture.d_lights, 1, d_framebuffer, params);
	CUDA_CHECK(cudaGetLastError());
	std::vector<float> framebuffer(pixels * 4);
	CUDA_CHECK(cudaMemcpy(framebuffer.data(), d_framebuffer, framebuffer.size() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_framebuffer));
	double total = 0.0;
	for (int i = 0; i < pixels; i++) total += framebuffer[i * 4];
	return total / pixels;
}

int main() {
	int deviceCount = 0;
	const cudaError_t deviceError = cudaGetDeviceCount(&deviceCount);
	if (deviceError == cudaErrorNoDevice || deviceError == cudaErrorInsufficientDriver
		|| (deviceError == cudaSuccess && deviceCount == 0)) {
		std::fprintf(stderr, "Skipping CUDA lighting test: no CUDA device or driver\n");
		return 77;
	}
	CUDA_CHECK(deviceError);
	const float rayOffset = 0.0001f;
	CUDA_CHECK(cudaMemcpyToSymbol(g_rayOffset, &rayOffset, sizeof(rayOffset)));

	fixture_t fixture;
	const std::vector<float> baseline = Sample(fixture, 1.0f, 1.0f);
	const double reference = ReferenceIntegral(1.0);
	std::printf("Triangle lighting: GPU %.6f, numerical reference %.6f\n", Mean(baseline), reference);
	CheckNear("area integral", Mean(baseline), reference, 0.002);
	CheckNear("emission boost", Mean(Sample(fixture, 1.0f, 2.0f)), 2.0 * Mean(baseline), 1e-5);
	CheckNear("two-sided emitter", Mean(Sample(fixture, -1.0f, 1.0f)), reference, 0.002);
	CheckNear("distant emitter", Mean(Sample(fixture, 10.0f, 1.0f)), ReferenceIntegral(10.0), 0.00002);
	CheckNear("weighted light selection", Mean(Sample(fixture, 1.0f, 1.0f, true)), 2.0 * reference, 0.006);

	ptMaterial_t& material = fixture.materials[0];
	material.emissionTexture = 1;
	material.texTransform[0] = material.texTransform[4] = 0.0f;
	material.texTransform[2] = 0.25f;
	material.texTransform[5] = 0.5f;
	const double halfGray = std::pow(128.0 / 255.0, 2.2);
	CheckNear("emission texture", Mean(Sample(fixture, 1.0f, 1.0f)), Mean(baseline) * halfGray, 1e-5);
	material.texTransform[2] = 0.75f;
	CheckNear("emission texture transform", Mean(Sample(fixture, 1.0f, 1.0f)), Mean(baseline), 1e-5);
	material.texTransform[0] = material.texTransform[4] = 1.0f;
	material.texTransform[2] = material.texTransform[5] = 0.0f;
	fixture.vertices[0].texcoord[0] = fixture.vertices[2].texcoord[0] = 0.25f;
	fixture.vertices[1].texcoord[0] = 0.75f;
	for (int i = 0; i < 3; i++) fixture.vertices[i].texcoord[1] = 0.5f;
	CheckNear("interpolated emission texture", Mean(Sample(fixture, 1.0f, 1.0f)),
		ReferenceIntegral(1.0, false, true), 0.002);
	material.albedoTexture = 0;
	material.alphaTest = 0.5f;
	material.texTransform[0] = material.texTransform[4] = 0.0f;
	material.texTransform[5] = 0.5f;
	material.texTransform[2] = 0.25f;
	CheckNear("texture alpha mask", Mean(Sample(fixture, 1.0f, 1.0f)), 0.0, 1e-8);
	material.texTransform[2] = 0.75f;
	CheckNear("passing texture alpha", Mean(Sample(fixture, 1.0f, 1.0f)), Mean(baseline), 1e-5);
	material.albedo[3] = 0.4f;
	CheckNear("material alpha mask", Mean(Sample(fixture, 1.0f, 1.0f)), 0.0, 1e-8);
	material.albedo[3] = 1.0f;
	material.albedoTexture = material.emissionTexture = -1;
	material.texTransform[0] = material.texTransform[4] = 1.0f;
	material.texTransform[2] = material.texTransform[5] = 0.0f;
	material.useVertexColor = 1;
	fixture.vertices[0].color[3] = 0.0f;
	CheckNear("interpolated vertex alpha mask", Mean(Sample(fixture, 1.0f, 1.0f)), ReferenceIntegral(1.0, true), 0.002);

	material.useVertexColor = 0;
	material.alphaTest = 0.0f;
	fixture.node.bounds[0] = fixture.node.bounds[1] = -100.0f;
	fixture.node.bounds[3] = fixture.node.bounds[4] = 100.0f;
	fixture.node.bounds[5] = 0.5f;
	fixture.node.primitiveCount = 2;
	CheckNear("opaque blocker", Mean(Sample(fixture, 1.0f, 1.0f)), 0.0, 1e-8);
	fixture.triangles[1].flags = 0;
	fixture.materials[1].alphaTest = 0.5f;
	fixture.materials[1].useVertexColor = 1;
	for (int i = 3; i < 6; i++) fixture.vertices[i].color[3] = 0.2f;
	CheckNear("transparent vertex-alpha shadow", Mean(Sample(fixture, 1.0f, 1.0f)), Mean(baseline), 1e-5);
	for (int i = 3; i < 6; i++) fixture.vertices[i].color[3] = 1.0f;
	CheckNear("opaque vertex-alpha shadow", Mean(Sample(fixture, 1.0f, 1.0f)), 0.0, 1e-8);
	fixture.materials[1].alphaTest = 0.0f;
	fixture.materials[1].useVertexColor = 0;
	fixture.materials[1].transmission = 1.0f;
	fixture.materials[1].ior = 1.0f;
	for (int c = 0; c < 3; c++) fixture.materials[1].albedo[c] = 0.5f;
	CheckNear("colored transmission shadow", Mean(Sample(fixture, 1.0f, 1.0f)), 0.5 * Mean(baseline), 0.001);
	fixture.node.primitiveCount = 1;
	for (int c = 0; c < 3; c++) material.emission[c] = 0.4f;
	std::vector<float> helper = Helpers(fixture);
	if (!(helper[0] > 0.0f)) {
		std::fprintf(stderr, "distant emitter has zero selection probability\n");
		return 1;
	}
	CheckNear("hemisphere rejection", helper[1], 0.0, 1e-8);
	CheckNear("zero-boost selection", helper[2], 0.0, 1e-8);
	CheckNear("indirect emission boost", helper[3], 0.8, 1e-6);
	CheckNear("sampled emitter suppression", helper[4], 0.0, 1e-8);
	fixture.triangles[0].flags &= ~PT_TRI_AREA_LIGHT;
	helper = Helpers(fixture);
	CheckNear("unrepresented emitter remains visible", helper[4], 0.8, 1e-6);

	material.useVertexColor = 1;
	material.alphaTest = 0.5f;
	for (int i = 0; i < 3; i++) fixture.vertices[i].color[3] = 0.2f;
	helper = Helpers(fixture);
	CheckNear("indirect vertex alpha mask", helper[3], 0.0, 1e-8);
	CheckNear("shadow vertex alpha mask", helper[5], 1.0, 1e-8);
	for (int i = 0; i < 3; i++) fixture.vertices[i].color[3] = 1.0f;
	helper = Helpers(fixture);
	CheckNear("opaque alpha-tested shadow", helper[5], 0.0, 1e-8);

	float* d_projection = Allocate<float>(9);
	ProjectionChecks<<<1, 1>>>(fixture.d_textures, d_projection);
	CUDA_CHECK(cudaGetLastError());
	float projection[9];
	CUDA_CHECK(cudaMemcpy(projection, d_projection, sizeof(projection), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_projection));
	CheckNear("projected texture", projection[0], halfGray, 1e-5);
	CheckNear("projected texture matrix", projection[1], 1.0, 1e-6);
	CheckNear("projected texture repeat", projection[2], halfGray, 1e-5);
	CheckNear("projected texture clamp", projection[3], 1.0, 1e-6);
	CheckNear("projected texture border", projection[4], 0.0, 1e-8);
	CheckNear("projected texture border filtering", projection[5], std::pow(0.5, 2.2), 1e-5);
	CheckNear("untransformed light volume", projection[6], 0.0, 1e-8);
	CheckNear("falloff texture", projection[7], halfGray, 1e-5);
	CheckNear("no falloff texture", projection[8], 1.0, 1e-6);

	material.alphaTest = 0.0f;
	material.useVertexColor = 0;
	for (int c = 0; c < 3; c++) material.emission[c] = 1.0f;
	for (int i = 0; i < 3; i++) fixture.vertices[i].position[2] = 2.0f;
	for (int i = 3; i < 6; i++) fixture.vertices[i].position[2] = 0.0f;
	fixture.lights[0].position[2] = 2.0f;
	fixture.triangles[0].flags |= PT_TRI_AREA_LIGHT;
	fixture.triangles[1].flags = PT_TRI_OPAQUE;
	fixture.node.bounds[2] = 0.0f;
	fixture.node.bounds[5] = 2.0f;
	fixture.node.primitiveCount = 2;
	fixture.materials[1].transmission = 0.0f;
	for (int c = 0; c < 3; c++) fixture.materials[1].albedo[c] = 1.0f;
	fixture.materials[1].roughness = 1.0f;
	const double direct = TracePath(fixture, 2, 1, 0.0f, 1.0f);
	const double bounce = TracePath(fixture, 2, 0, 0.0f, 1.0f);
	std::printf("Path estimates: direct %.6f, bounce %.6f\n", direct, bounce);
	CheckNear("path direct versus bounce", direct, bounce, 0.006);
	CheckNear("path emission boost", TracePath(fixture, 2, 1, 0.0f, 2.0f), direct * 2.0, 1e-5);
	CheckNear("single-depth indirect emitter", TracePath(fixture, 1, 0, 1.0f, 1.0f), bounce, 0.006);
	CheckNear("single-depth sampled emitter suppression", TracePath(fixture, 1, 1, 1.0f, 1.0f), direct, 1e-5);
	fixture.triangles[0].flags &= ~PT_TRI_AREA_LIGHT;
	CheckNear("unrepresented path emitter", TracePath(fixture, 2, 1, 0.0f, 1.0f), direct + bounce, 0.006);

	std::puts("CUDA lighting regression checks passed");
	return 0;
}
