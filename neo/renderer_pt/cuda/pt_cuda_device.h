#ifndef __PT_CUDA_DEVICE_H__
#define __PT_CUDA_DEVICE_H__

/*
===============================================================================

	CUDA backend internals, shared by the .cu kernels and pt_backend_cuda.cpp.
	Nothing outside renderer_pt/cuda/ should include this.

===============================================================================
*/

#include <cuda_runtime.h>

#include "renderer_pt/pt_types.h"

#define TILE_SIZE		16
#define MAX_TEXTURES	PT_MAX_TEXTURES

// float constant, M_PI is a double and silently promotes device math to FP64
#define PT_PI			3.14159265358979f

// device-side texture descriptor, built by the backend from ptTextureInfo_t
struct cudaTexture_t {
	int width;
	int height;
	unsigned char *data;		// RGBA8, NULL when not resident
};

struct ptRay_t {
	float origin[3];
	float direction[3];
	float tMin;
	float tMax;
};

struct ptHitInfo_t {
	bool hit;
	float t;
	float position[3];
	float normal[3];
	float tangent[3];
	float binormal[3];
	float texcoord[2];
	float barycentrics[3];		// (w, u, v)
	int materialIndex;
	int triangleIndex;
};

#ifdef __CUDACC__
// Expands ptRenderParams_t into the local names the kernels were written with.
#define PT_UNPACK_RENDER_PARAMS( p ) \
	[[maybe_unused]] const int width = (p).width; \
	[[maybe_unused]] const int height = (p).height; \
	[[maybe_unused]] const float *cameraPos = (p).camPos; \
	[[maybe_unused]] const float *cameraForward = (p).camForward; \
	[[maybe_unused]] const float *cameraRight = (p).camRight; \
	[[maybe_unused]] const float *cameraUp = (p).camUp; \
	[[maybe_unused]] const float fov = (p).fovX; \
	[[maybe_unused]] const float fovY = (p).fovY; \
	[[maybe_unused]] const int samplesPerPixel = (p).samplesPerPixel; \
	[[maybe_unused]] const int maxDepth = (p).maxDepth; \
	[[maybe_unused]] const int maxLightSamples = (p).maxLightSamples; \
	[[maybe_unused]] const int frameIndex = (p).frameIndex; \
	[[maybe_unused]] const float emissionBoost = (p).emissionBoost; \
	[[maybe_unused]] const float indirectProb = (p).indirectProb; \
	[[maybe_unused]] const int rrEnabled = (p).rrEnabled; \
	[[maybe_unused]] const int rrMinBounces = (p).rrMinBounces; \
	[[maybe_unused]] const float rrSurvivalMin = (p).rrSurvivalMin; \
	[[maybe_unused]] const float earlyTermThreshold = (p).earlyTermThreshold; \
	[[maybe_unused]] const float fireflyClamp = (p).fireflyClamp; \
	[[maybe_unused]] const float throughputClamp = (p).throughputClamp; \
	[[maybe_unused]] const float rayOffset = (p).rayOffset; \
	[[maybe_unused]] const float specularBoost = (p).specularBoost; \
	[[maybe_unused]] const float skyIntensity = (p).skyIntensity; \
	[[maybe_unused]] const float *skyColorZenith = (p).skyZenith; \
	[[maybe_unused]] const float *skyColorHorizon = (p).skyHorizon; \
	[[maybe_unused]] const float *skyColorGround = (p).skyGround; \
	[[maybe_unused]] const float volumetricDensity = (p).volDensity; \
	[[maybe_unused]] const int volumetricSteps = (p).volSteps; \
	[[maybe_unused]] const float volumetricAnisotropy = (p).volAnisotropy; \
	[[maybe_unused]] const float volFalloff = (p).volFalloff; \
	[[maybe_unused]] const float volMaxDist = (p).volMaxDist
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const ptVertex_t *		vertices;
	const ptTriangle_t *	triangles;
	const ptBVHNode_t *		bvhNodes;
	const int *				triIndices;
	const ptMaterial_t *	materials;
	const cudaTexture_t *	textures;
	const ptLight_t *		lights;
	int						numLights;
} ptCudaSceneView_t;

// traces params.width x params.height into the float4 accumulation buffer
// (row pitch == params.width)
void LaunchPathTracingKernel( const ptCudaSceneView_t *scene, float *accumBuffer,
							  const ptRenderParams_t *params, cudaStream_t stream );

// resolves the accumulation buffer into RGBA8
void LaunchToneMappingKernel( const float *hdrBuffer, unsigned char *ldrBuffer, int width, int height,
							  float exposure, float gamma, int toneMapMode, int frameCount, cudaStream_t stream );

// GPU LBVH construction, sceneBounds is a host pointer (min xyz, max xyz)
void LaunchBuildLBVH( const ptTriangle_t *d_triangles, int numTriangles, ptBVHNode_t *d_bvhNodes, int *d_triIndices,
					  unsigned int *d_mortonCodes, int *d_sortedIndices, int *d_parents, int *d_atomicCounters,
					  const float *sceneBounds, cudaStream_t stream );

#ifdef __cplusplus
}
#endif

#endif /* !__PT_CUDA_DEVICE_H__ */
