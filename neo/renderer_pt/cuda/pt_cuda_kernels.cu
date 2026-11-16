#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>

#include "renderer_pt/cuda/pt_cuda_device.h"
#include "renderer_pt/cuda/pt_kernel_triangles.cuh"
#include "renderer_pt/cuda/pt_kernel_triangles_textured.cuh"
#include "renderer_pt/cuda/pt_kernel_triangles_raytraced.cuh"
#include "renderer_pt/cuda/pt_kernel_pathtracing.cuh"
#include "renderer_pt/cuda/pt_kernel_denoising.cuh"

/*
========================
LaunchPathTracingKernel

All render parameters are passed by value. Never pass host pointers to a
kernel: they are only dereferenceable on HMM/ATS capable systems.
========================
*/
extern "C" void LaunchPathTracingKernel( const ptCudaSceneView_t *scene, float *accumBuffer,
										 float *radianceBuffer, ptCudaDenoiseGuide_t *guideBuffer,
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
				accumBuffer, p, radianceBuffer, guideBuffer );
			break;
	}
}

extern "C" cudaError_t LaunchDenoisingKernel( const ptCudaDenoiseBuffers_t *buffers, const float *accumBuffer, float *filteredBuffer,
											 const ptRenderParams_t *params, const ptRenderParams_t *previousParams,
											 int frameSlot, int historyValid, cudaStream_t stream ) {
	const ptRenderParams_t p = *params;
	const ptCudaDenoiseBuffers_t &b = *buffers;
	const int prev = frameSlot ^ 1;
	const dim3 blockSize( TILE_SIZE, TILE_SIZE );
	const dim3 gridSize( ( p.width + TILE_SIZE - 1 ) / TILE_SIZE, ( p.height + TILE_SIZE - 1 ) / TILE_SIZE );
	SVGFTemporalKernel<<<gridSize, blockSize, 0, stream>>>(
		b.radiance, b.guides[frameSlot], b.guides[prev], b.history[prev], b.moments[prev],
		filteredBuffer, b.moments[frameSlot], p, *previousParams, historyValid );
	cudaError_t error = cudaGetLastError();
	if ( error != cudaSuccess ) return error;

	const float *input = filteredBuffer;
	for ( int iteration = 0; iteration < 4; iteration++ ) {
		float *output = iteration == 0 ? b.history[frameSlot] : ( iteration == 2 ? filteredBuffer : b.scratch );
		SVGFAtrousKernel<<<gridSize, blockSize, 0, stream>>>(
			input, b.guides[frameSlot], output, p.width, p.height, 1 << iteration, p.denoiseStrength );
		error = cudaGetLastError();
		if ( error != cudaSuccess ) return error;
		input = output;
	}
	SVGFResolveKernel<<<gridSize, blockSize, 0, stream>>>(
		input, accumBuffer, b.guides[frameSlot], filteredBuffer, p.width, p.height );
	return cudaGetLastError();
}

/*
========================
LaunchToneMappingKernel
========================
*/
extern "C" void LaunchToneMappingKernel( const float *hdrBuffer, unsigned char *ldrBuffer, int width, int height,
										 float exposure, float gamma, int toneMapMode, cudaStream_t stream ) {
	const dim3 blockSize( TILE_SIZE, TILE_SIZE );
	const dim3 gridSize( ( width + TILE_SIZE - 1 ) / TILE_SIZE, ( height + TILE_SIZE - 1 ) / TILE_SIZE );

	ToneMappingKernel<<<gridSize, blockSize, 0, stream>>>( hdrBuffer, ldrBuffer, width, height, exposure, gamma, toneMapMode );
}
