#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_kernel_triangles.cuh"
#include "renderer_pt/cuda/pt_kernel_triangles_textured.cuh"
#include "renderer_pt/cuda/pt_kernel_triangles_raytraced.cuh"
#include "renderer_pt/cuda/pt_kernel_pathtracing.cuh"

/*
========================
LaunchPathTracingKernel

All render parameters are passed by value. Never pass host pointers to a
kernel: they are only dereferenceable on HMM/ATS capable systems.
========================
*/
extern "C" void LaunchPathTracingKernel( const ptCudaSceneView_t *scene, float *accumBuffer,
										 const ptRenderParams_t *params, cudaStream_t stream ) {
	const ptRenderParams_t p = *params;

	// staged from pageable memory before the call returns, so the stack sources are fine
	cudaMemcpyToSymbolAsync( g_softShadowScale, &p.softShadowScale, sizeof( float ), 0, cudaMemcpyHostToDevice, stream );
	cudaMemcpyToSymbolAsync( g_rayOffset, &p.rayOffset, sizeof( float ), 0, cudaMemcpyHostToDevice, stream );

	const dim3 blockSize( TILE_SIZE, TILE_SIZE );
	const dim3 gridSize( ( p.width + TILE_SIZE - 1 ) / TILE_SIZE, ( p.height + TILE_SIZE - 1 ) / TILE_SIZE );

	const ptCudaSceneView_t &s = *scene;

	switch ( p.renderMode ) {
		case PT_MODE_BVH_DEBUG:
			TriangleDrawKernel<<<gridSize, blockSize, 0, stream>>>(
				s.vertices, s.triangles, s.bvhNodes, s.triIndices, s.materials, s.textures, s.lights, s.numLights,
				accumBuffer, p );
			break;

		case PT_MODE_TEX_COMBINED:
		case PT_MODE_TEX_BASECOLOR:
		case PT_MODE_TEX_ALBEDO:
		case PT_MODE_TEX_NORMAL:
		case PT_MODE_TEX_SPECULAR:
			// 0 = combined, 1 = base color, 2 = albedo texture, 3 = normal map, 4 = specular map
			TriangleDrawTexturedKernel<<<gridSize, blockSize, 0, stream>>>(
				s.vertices, s.triangles, s.bvhNodes, s.triIndices, s.materials, s.textures, s.lights, s.numLights,
				accumBuffer, p.renderMode - PT_MODE_TEX_COMBINED, p );
			break;

		case PT_MODE_WHITTED:
			TriangleDrawRayTracedKernel<<<gridSize, blockSize, 0, stream>>>(
				s.vertices, s.triangles, s.bvhNodes, s.triIndices, s.materials, s.textures, s.lights, s.numLights,
				accumBuffer, p );
			break;

		default:
			PathTracingKernel<<<gridSize, blockSize, 0, stream>>>(
				s.vertices, s.triangles, s.bvhNodes, s.triIndices, s.materials, s.textures, s.lights, s.numLights,
				accumBuffer, p );
			break;
	}
}

/*
========================
LaunchToneMappingKernel
========================
*/
extern "C" void LaunchToneMappingKernel( const float *hdrBuffer, unsigned char *ldrBuffer, int width, int height,
										 float exposure, float gamma, int toneMapMode, int frameCount, cudaStream_t stream ) {
	const dim3 blockSize( TILE_SIZE, TILE_SIZE );
	const dim3 gridSize( ( width + TILE_SIZE - 1 ) / TILE_SIZE, ( height + TILE_SIZE - 1 ) / TILE_SIZE );

	ToneMappingKernel<<<gridSize, blockSize, 0, stream>>>( hdrBuffer, ldrBuffer, width, height, exposure, gamma, toneMapMode, frameCount );
}
