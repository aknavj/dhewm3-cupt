#ifndef __PT_BACKEND_H__
#define __PT_BACKEND_H__

#include <stddef.h>

#include "renderer_pt/pt_types.h"

class idPathTracerScene;

typedef struct {
	float		traceMs;			// GPU time of the last Render()
	int			numVertices;
	int			numTriangles;
	int			numBVHNodes;
	int			numMaterials;
	int			numLights;
	int			numTextures;
	size_t		deviceBytes;		// approximate device memory owned by the backend
} ptBackendStats_t;

/*
===============================================================================

	idPathTracerBackend

	Abstract compute device. The API-agnostic front end (pt_render.cpp) builds
	an idPathTracerScene on the CPU and hands it to a backend which owns all
	device memory, the acceleration structure, the accumulation buffer and the
	kernels. Adding a new compute API (OpenCL, Vulkan compute, CPU, ...) means
	subclassing this and registering a factory in pt_render.cpp.

	Call order per view:
		UploadScene( scene )  ->  Render( params )  ->  ReadPixels( ... )

	All methods are called from the render back end thread.

===============================================================================
*/
class idPathTracerBackend {
public:
	virtual					~idPathTracerBackend() {}

	virtual const char *	Name() const = 0;
	virtual void			PrintInfo() const = 0;

	// maxWidth / maxHeight is the largest trace resolution Render() will be asked for
	virtual bool			Init( int maxWidth, int maxHeight ) = 0;
	virtual void			Shutdown() = 0;
	virtual bool			Resize( int maxWidth, int maxHeight ) = 0;

	// false after an unrecoverable device error; the front end will fall back to OpenGL
	virtual bool			IsHealthy() const = 0;

	// uploads geometry, materials, lights and the scene's pending textures, then
	// rebuilds the acceleration structure
	virtual bool			UploadScene( const idPathTracerScene &scene ) = 0;

	// traces params.width x params.height pixels into the accumulation buffer and
	// tone maps the result into an RGBA8 image
	virtual bool			Render( const ptRenderParams_t &params ) = 0;

	// copies the last rendered RGBA8 image (width x height of the last Render) to dest
	virtual bool			ReadPixels( unsigned char *dest, int width, int height ) = 0;

	virtual void			ResetAccumulation() = 0;
	virtual int				GetAccumulatedFrames() const = 0;

	virtual void			GetStats( ptBackendStats_t &stats ) const = 0;
};

/*
===============================================================================

	Backend registry. Each compiled-in backend exposes an availability probe and
	a factory; pt_render.cpp lists them in a table selected by r_ptBackend.

===============================================================================
*/
typedef struct {
	const char *			name;
	bool					( *isAvailable )();
	idPathTracerBackend *	( *create )();
} ptBackendFactory_t;

#ifdef HAVE_CUDA
bool						PT_CUDA_IsAvailable();
idPathTracerBackend *		PT_CUDA_CreateBackend();
#endif

#endif /* !__PT_BACKEND_H__ */
