#include "sys/platform.h"
#include "framework/Common.h"
#include "idlib/math/Math.h"

#include "renderer_pt/pt_backend.h"
#include "renderer_pt/pt_scene.h"
#include "renderer_pt/cuda/pt_cuda_device.h"

/*
===============================================================================

	idPathTracerBackendCUDA

	All device work is ordered on a single stream. The only host
	synchronization point is the end of Render(), so ReadPixels() and the
	timing queries see finished results.

===============================================================================
*/

#define PT_CUDA_CHECK( call ) \
	do { \
		const cudaError_t _err = ( call ); \
		if ( _err != cudaSuccess ) { \
			OnError( #call, _err ); \
			return false; \
		} \
	} while ( 0 )

class idPathTracerBackendCUDA : public idPathTracerBackend {
public:
							idPathTracerBackendCUDA();
	virtual					~idPathTracerBackendCUDA();

	static bool				IsAvailable();

	virtual const char *	Name() const { return "cuda"; }
	virtual void			PrintInfo() const;

	virtual bool			Init( int maxWidth, int maxHeight );
	virtual void			Shutdown();
	virtual bool			Resize( int maxWidth, int maxHeight );
	virtual bool			IsHealthy() const { return healthy; }

	virtual bool			UploadScene( const idPathTracerScene &scene );
	virtual bool			Render( const ptRenderParams_t &params );
	virtual bool			ReadPixels( unsigned char *dest, int width, int height );

	virtual void			ResetAccumulation() { accumFrames = 0; }
	virtual int				GetAccumulatedFrames() const { return accumFrames; }
	virtual void			GetStats( ptBackendStats_t &stats ) const;

private:
	template< class T >
	bool					Reserve( T *&ptr, int &capacity, int needed, const char *what );
	bool					UploadTextures( const idPathTracerScene &scene );
	void					FreeTextures();
	bool					BuildBVH( const idPathTracerScene &scene );
	void					OnError( const char *what, cudaError_t err );

	bool					initialized;
	bool					healthy;
	cudaStream_t			stream;
	cudaEvent_t				timerStart;
	cudaEvent_t				timerStop;

	// output
	int						maxWidth;
	int						maxHeight;
	float *					d_accum;			// float4 HDR accumulation, row pitch == trace width
	unsigned char *			d_output;			// RGBA8
	int						lastWidth;
	int						lastHeight;
	int						accumFrames;
	float					traceMs;

	// scene, all grown on demand
	ptVertex_t *			d_vertices;		int capVertices;
	ptTriangle_t *			d_triangles;	int capTriangles;
	ptBVHNode_t *			d_bvhNodes;		int capBVHNodes;
	int *					d_triIndices;	int capTriIndices;
	unsigned int *			d_mortonCodes;	int capMorton;
	unsigned int *			d_mortonAlt;	int capMortonAlt;
	unsigned char *			d_sortTemp;		int capSortTemp;	// CUB radix sort scratch, bytes
	int *					d_sortedIndices;int capSorted;
	int *					d_parents;		int capParents;
	int *					d_atomics;		int capAtomics;
	ptMaterial_t *			d_materials;	int capMaterials;
	ptLight_t *				d_lights;		int capLights;

	int						numVertices;
	int						numTriangles;
	int						numBVHNodes;
	int						numMaterials;
	int						numLights;

	// textures: host mirror of the device descriptor table
	cudaTexture_t			h_texDesc[PT_MAX_TEXTURES];
	cudaTexture_t *			d_texDesc;
	int						numTextures;
	int						textureGeneration;
	size_t					textureBytes;
};

/*
========================
idPathTracerBackendCUDA::idPathTracerBackendCUDA
========================
*/
idPathTracerBackendCUDA::idPathTracerBackendCUDA() {
	initialized = false;
	healthy = false;
	stream = NULL;
	timerStart = NULL;
	timerStop = NULL;

	maxWidth = maxHeight = 0;
	d_accum = NULL;
	d_output = NULL;
	lastWidth = lastHeight = 0;
	accumFrames = 0;
	traceMs = 0.0f;

	d_vertices = NULL;		capVertices = 0;
	d_triangles = NULL;		capTriangles = 0;
	d_bvhNodes = NULL;		capBVHNodes = 0;
	d_triIndices = NULL;	capTriIndices = 0;
	d_mortonCodes = NULL;	capMorton = 0;
	d_mortonAlt = NULL;		capMortonAlt = 0;
	d_sortTemp = NULL;		capSortTemp = 0;
	d_sortedIndices = NULL;	capSorted = 0;
	d_parents = NULL;		capParents = 0;
	d_atomics = NULL;		capAtomics = 0;
	d_materials = NULL;		capMaterials = 0;
	d_lights = NULL;		capLights = 0;

	numVertices = numTriangles = numBVHNodes = numMaterials = numLights = 0;

	memset( h_texDesc, 0, sizeof( h_texDesc ) );
	d_texDesc = NULL;
	numTextures = 0;
	textureGeneration = -1;
	textureBytes = 0;
}

/*
========================
idPathTracerBackendCUDA::~idPathTracerBackendCUDA
========================
*/
idPathTracerBackendCUDA::~idPathTracerBackendCUDA() {
	Shutdown();
}

/*
========================
idPathTracerBackendCUDA::IsAvailable
========================
*/
bool idPathTracerBackendCUDA::IsAvailable() {
	int deviceCount = 0;
	if ( cudaGetDeviceCount( &deviceCount ) != cudaSuccess || deviceCount == 0 ) {
		cudaGetLastError();		// clear the sticky "no device" error
		return false;
	}
	cudaDeviceProp prop;
	if ( cudaGetDeviceProperties( &prop, 0 ) != cudaSuccess ) {
		return false;
	}
	return prop.major >= 3;
}

/*
========================
idPathTracerBackendCUDA::OnError
========================
*/
void idPathTracerBackendCUDA::OnError( const char *what, cudaError_t err ) {
	common->Warning( "CUDA path tracer: %s failed: %s\n", what, cudaGetErrorString( err ) );
	healthy = false;
}

/*
========================
idPathTracerBackendCUDA::PrintInfo
========================
*/
void idPathTracerBackendCUDA::PrintInfo() const {
	int deviceCount = 0;
	cudaGetDeviceCount( &deviceCount );
	int runtime = 0, driver = 0;
	cudaRuntimeGetVersion( &runtime );
	cudaDriverGetVersion( &driver );

	common->Printf( "CUDA runtime %d.%d, driver %d.%d, %d device(s)\n",
		runtime / 1000, ( runtime % 1000 ) / 10, driver / 1000, ( driver % 1000 ) / 10, deviceCount );
	for ( int i = 0; i < deviceCount; i++ ) {
		cudaDeviceProp prop;
		if ( cudaGetDeviceProperties( &prop, i ) != cudaSuccess ) {
			continue;
		}
		common->Printf( "  %d: %s, sm_%d%d, %d SMs, %.2f GB%s\n", i, prop.name, prop.major, prop.minor,
			prop.multiProcessorCount, prop.totalGlobalMem / ( 1024.0 * 1024.0 * 1024.0 ), i == 0 ? " (active)" : "" );
	}

	ptBackendStats_t stats;
	GetStats( stats );
	common->Printf( "  output %dx%d, %d verts, %d tris, %d BVH nodes, %d materials, %d lights, %d textures\n",
		maxWidth, maxHeight, stats.numVertices, stats.numTriangles, stats.numBVHNodes,
		stats.numMaterials, stats.numLights, stats.numTextures );
	common->Printf( "  device memory %.1f MB, last trace %.2f ms, %d accumulated frames\n",
		stats.deviceBytes / ( 1024.0 * 1024.0 ), traceMs, accumFrames );
}

/*
========================
idPathTracerBackendCUDA::Init
========================
*/
bool idPathTracerBackendCUDA::Init( int width, int height ) {
	Shutdown();

	if ( !IsAvailable() ) {
		common->Warning( "CUDA path tracer: no compatible CUDA device found\n" );
		return false;
	}

	healthy = true;
	initialized = true;

	PT_CUDA_CHECK( cudaSetDevice( 0 ) );
	PT_CUDA_CHECK( cudaStreamCreate( &stream ) );
	PT_CUDA_CHECK( cudaEventCreate( &timerStart ) );
	PT_CUDA_CHECK( cudaEventCreate( &timerStop ) );

	PT_CUDA_CHECK( cudaMalloc( &d_texDesc, PT_MAX_TEXTURES * sizeof( cudaTexture_t ) ) );
	PT_CUDA_CHECK( cudaMemset( d_texDesc, 0, PT_MAX_TEXTURES * sizeof( cudaTexture_t ) ) );

	if ( !Resize( width, height ) ) {
		return false;
	}

	cudaDeviceProp prop;
	PT_CUDA_CHECK( cudaGetDeviceProperties( &prop, 0 ) );
	common->Printf( "CUDA path tracer: %s (sm_%d%d), %dx%d\n", prop.name, prop.major, prop.minor, width, height );
	return true;
}

/*
========================
idPathTracerBackendCUDA::Shutdown

Frees only what this backend owns. Never cudaDeviceReset(): it tears down
the context for every other CUDA user in the process.
========================
*/
void idPathTracerBackendCUDA::Shutdown() {
	if ( !initialized ) {
		return;
	}
	if ( stream ) {
		cudaStreamSynchronize( stream );
	}

	FreeTextures();

	cudaFree( d_texDesc );			d_texDesc = NULL;
	cudaFree( d_accum );			d_accum = NULL;
	cudaFree( d_output );			d_output = NULL;
	cudaFree( d_vertices );			d_vertices = NULL;		capVertices = 0;
	cudaFree( d_triangles );		d_triangles = NULL;		capTriangles = 0;
	cudaFree( d_bvhNodes );			d_bvhNodes = NULL;		capBVHNodes = 0;
	cudaFree( d_triIndices );		d_triIndices = NULL;	capTriIndices = 0;
	cudaFree( d_mortonCodes );		d_mortonCodes = NULL;	capMorton = 0;
	cudaFree( d_mortonAlt );		d_mortonAlt = NULL;		capMortonAlt = 0;
	cudaFree( d_sortTemp );			d_sortTemp = NULL;		capSortTemp = 0;
	d_mortonAlt = NULL;		capMortonAlt = 0;
	d_sortTemp = NULL;		capSortTemp = 0;
	cudaFree( d_sortedIndices );	d_sortedIndices = NULL;	capSorted = 0;
	cudaFree( d_parents );			d_parents = NULL;		capParents = 0;
	cudaFree( d_atomics );			d_atomics = NULL;		capAtomics = 0;
	cudaFree( d_materials );		d_materials = NULL;		capMaterials = 0;
	cudaFree( d_lights );			d_lights = NULL;		capLights = 0;

	if ( timerStart ) {
		cudaEventDestroy( timerStart );
		timerStart = NULL;
	}
	if ( timerStop ) {
		cudaEventDestroy( timerStop );
		timerStop = NULL;
	}
	if ( stream ) {
		cudaStreamDestroy( stream );
		stream = NULL;
	}

	numVertices = numTriangles = numBVHNodes = numMaterials = numLights = 0;
	maxWidth = maxHeight = lastWidth = lastHeight = 0;
	accumFrames = 0;
	textureGeneration = -1;
	initialized = false;
	healthy = false;
}

/*
========================
idPathTracerBackendCUDA::Resize
========================
*/
bool idPathTracerBackendCUDA::Resize( int width, int height ) {
	if ( !initialized ) {
		return false;
	}
	width = Max( width, 1 );
	height = Max( height, 1 );
	if ( width == maxWidth && height == maxHeight && d_accum && d_output ) {
		return true;
	}
	if ( stream ) {
		cudaStreamSynchronize( stream );
	}

	cudaFree( d_accum );	d_accum = NULL;
	cudaFree( d_output );	d_output = NULL;
	maxWidth = maxHeight = 0;

	const size_t pixels = (size_t)width * height;
	PT_CUDA_CHECK( cudaMalloc( &d_accum, pixels * 4 * sizeof( float ) ) );
	PT_CUDA_CHECK( cudaMalloc( &d_output, pixels * 4 ) );

	maxWidth = width;
	maxHeight = height;
	lastWidth = lastHeight = 0;
	accumFrames = 0;
	return true;
}

/*
========================
idPathTracerBackendCUDA::Reserve
========================
*/
template< class T >
bool idPathTracerBackendCUDA::Reserve( T *&ptr, int &capacity, int needed, const char *what ) {
	if ( needed <= capacity ) {
		return true;
	}
	const int newCapacity = Max( needed, capacity + capacity / 2 );

	cudaFree( ptr );
	ptr = NULL;
	capacity = 0;

	const cudaError_t err = cudaMalloc( &ptr, (size_t)newCapacity * sizeof( T ) );
	if ( err != cudaSuccess ) {
		ptr = NULL;
		common->Warning( "CUDA path tracer: failed to allocate %d %s (%.1f MB): %s\n",
			newCapacity, what, (double)newCapacity * sizeof( T ) / ( 1024.0 * 1024.0 ), cudaGetErrorString( err ) );
		healthy = false;
		return false;
	}
	capacity = newCapacity;
	return true;
}

/*
========================
idPathTracerBackendCUDA::FreeTextures
========================
*/
void idPathTracerBackendCUDA::FreeTextures() {
	for ( int i = 0; i < PT_MAX_TEXTURES; i++ ) {
		if ( h_texDesc[i].data ) {
			cudaFree( h_texDesc[i].data );
		}
	}
	memset( h_texDesc, 0, sizeof( h_texDesc ) );
	numTextures = 0;
	textureBytes = 0;
}

/*
========================
idPathTracerBackendCUDA::UploadTextures
========================
*/
bool idPathTracerBackendCUDA::UploadTextures( const idPathTracerScene &scene ) {
	if ( scene.TextureGeneration() != textureGeneration ) {
		cudaStreamSynchronize( stream );	// kernels may still sample the old textures
		FreeTextures();
		textureGeneration = scene.TextureGeneration();
	}

	const idList<ptTextureUpload_t> &pending = scene.PendingTextures();
	for ( int i = 0; i < pending.Num(); i++ ) {
		const ptTextureUpload_t &up = pending[i];
		if ( up.index < 0 || up.index >= PT_MAX_TEXTURES || !up.rgba ) {
			continue;
		}
		cudaTexture_t &desc = h_texDesc[up.index];
		const size_t bytes = (size_t)up.width * up.height * 4;

		if ( desc.data && ( desc.width != up.width || desc.height != up.height ) ) {
			cudaStreamSynchronize( stream );
			cudaFree( desc.data );
			textureBytes -= (size_t)desc.width * desc.height * 4;
			desc.data = NULL;
		}
		if ( !desc.data ) {
			const cudaError_t err = cudaMalloc( &desc.data, bytes );
			if ( err != cudaSuccess ) {
				// not fatal, the sampler returns white for missing textures
				common->Warning( "CUDA path tracer: texture %d (%dx%d) allocation failed: %s\n",
					up.index, up.width, up.height, cudaGetErrorString( err ) );
				cudaGetLastError();
				desc.data = NULL;
				desc.width = desc.height = 0;
				continue;
			}
			textureBytes += bytes;
		}
		desc.width = up.width;
		desc.height = up.height;
		PT_CUDA_CHECK( cudaMemcpyAsync( desc.data, up.rgba, bytes, cudaMemcpyHostToDevice, stream ) );
	}

	numTextures = Min( scene.Textures().Num(), PT_MAX_TEXTURES );
	if ( pending.Num() > 0 || numTextures == 0 ) {
		// the whole table, so flushed slots are cleared on the device as well
		PT_CUDA_CHECK( cudaMemcpyAsync( d_texDesc, h_texDesc, sizeof( h_texDesc ), cudaMemcpyHostToDevice, stream ) );
	}
	return true;
}

/*
========================
idPathTracerBackendCUDA::BuildBVH
========================
*/
bool idPathTracerBackendCUDA::BuildBVH( const idPathTracerScene &scene ) {
	const int n = numTriangles;
	numBVHNodes = 0;
	if ( n == 0 ) {
		return true;
	}

	if ( !Reserve( d_bvhNodes, capBVHNodes, 2 * n - 1, "BVH nodes" ) ||
		 !Reserve( d_triIndices, capTriIndices, n, "BVH indices" ) ||
		 !Reserve( d_mortonCodes, capMorton, n, "morton codes" ) ||
		 !Reserve( d_mortonAlt, capMortonAlt, n, "morton codes" ) ||
		 !Reserve( d_sortedIndices, capSorted, n, "sort indices" ) ||
		 !Reserve( d_sortTemp, capSortTemp, (int)LBVHSortTempBytes( n ), "sort scratch bytes" ) ||
		 !Reserve( d_parents, capParents, 2 * n, "BVH parents" ) ||
		 !Reserve( d_atomics, capAtomics, n, "BVH counters" ) ) {
		return false;
	}

	float bounds[6];
	memcpy( bounds, scene.Bounds(), sizeof( bounds ) );
	for ( int a = 0; a < 3; a++ ) {
		if ( bounds[a + 3] - bounds[a] < 0.01f ) {
			bounds[a] -= 0.5f;
			bounds[a + 3] += 0.5f;
		}
	}

	LaunchBuildLBVH( d_triangles, n, d_bvhNodes, d_triIndices, d_mortonCodes, d_mortonAlt, d_sortedIndices,
		d_sortTemp, (size_t)capSortTemp, d_parents, d_atomics, bounds, stream );
	PT_CUDA_CHECK( cudaGetLastError() );

	numBVHNodes = 2 * n - 1;
	return true;
}

/*
========================
idPathTracerBackendCUDA::UploadScene
========================
*/
bool idPathTracerBackendCUDA::UploadScene( const idPathTracerScene &scene ) {
	if ( !healthy ) {
		return false;
	}

	numVertices = scene.Vertices().Num();
	numTriangles = scene.Triangles().Num();
	numMaterials = scene.Materials().Num();
	numLights = scene.Lights().Num();

	if ( !Reserve( d_vertices, capVertices, Max( numVertices, 1 ), "vertices" ) ||
		 !Reserve( d_triangles, capTriangles, Max( numTriangles, 1 ), "triangles" ) ||
		 !Reserve( d_materials, capMaterials, Max( numMaterials, 1 ), "materials" ) ||
		 !Reserve( d_lights, capLights, Max( numLights, 1 ), "lights" ) ) {
		numVertices = numTriangles = numMaterials = numLights = 0;
		return false;
	}

	if ( numVertices > 0 ) {
		PT_CUDA_CHECK( cudaMemcpyAsync( d_vertices, scene.Vertices().Ptr(), numVertices * sizeof( ptVertex_t ), cudaMemcpyHostToDevice, stream ) );
	}
	if ( numTriangles > 0 ) {
		PT_CUDA_CHECK( cudaMemcpyAsync( d_triangles, scene.Triangles().Ptr(), numTriangles * sizeof( ptTriangle_t ), cudaMemcpyHostToDevice, stream ) );
	}
	if ( numMaterials > 0 ) {
		PT_CUDA_CHECK( cudaMemcpyAsync( d_materials, scene.Materials().Ptr(), numMaterials * sizeof( ptMaterial_t ), cudaMemcpyHostToDevice, stream ) );
	}
	if ( numLights > 0 ) {
		PT_CUDA_CHECK( cudaMemcpyAsync( d_lights, scene.Lights().Ptr(), numLights * sizeof( ptLight_t ), cudaMemcpyHostToDevice, stream ) );
	}

	if ( !UploadTextures( scene ) ) {
		return false;
	}
	return BuildBVH( scene );
}

/*
========================
idPathTracerBackendCUDA::Render
========================
*/
bool idPathTracerBackendCUDA::Render( const ptRenderParams_t &params ) {
	if ( !healthy || !d_accum || !d_output ) {
		return false;
	}

	ptRenderParams_t p = params;
	p.width = idMath::ClampInt( 1, maxWidth, p.width );
	p.height = idMath::ClampInt( 1, maxHeight, p.height );

	if ( p.width != lastWidth || p.height != lastHeight ) {
		accumFrames = 0;
	}
	lastWidth = p.width;
	lastHeight = p.height;

	const size_t pixels = (size_t)p.width * p.height;

	if ( numTriangles == 0 ) {
		accumFrames = 0;
		PT_CUDA_CHECK( cudaMemsetAsync( d_output, 0, pixels * 4, stream ) );
		PT_CUDA_CHECK( cudaStreamSynchronize( stream ) );
		return true;
	}

	// debug modes overwrite the buffer every frame, only path tracing accumulates
	const bool accumulate = ( p.renderMode == PT_MODE_PATHTRACE );
	if ( !accumulate ) {
		accumFrames = 0;
	}
	if ( accumFrames == 0 ) {
		PT_CUDA_CHECK( cudaMemsetAsync( d_accum, 0, pixels * 4 * sizeof( float ), stream ) );
	}
	p.frameIndex = accumFrames;

	ptCudaSceneView_t view;
	view.vertices = d_vertices;
	view.triangles = d_triangles;
	view.bvhNodes = d_bvhNodes;
	view.triIndices = d_triIndices;
	view.materials = d_materials;
	view.textures = d_texDesc;
	view.lights = d_lights;
	view.numLights = numLights;

	cudaEventRecord( timerStart, stream );

	LaunchPathTracingKernel( &view, d_accum, &p, stream );
	PT_CUDA_CHECK( cudaGetLastError() );

	accumFrames++;
	LaunchToneMappingKernel( d_accum, d_output, p.width, p.height, p.exposure, p.gamma, p.toneMapMode,
		stream );
	PT_CUDA_CHECK( cudaGetLastError() );

	cudaEventRecord( timerStop, stream );
	PT_CUDA_CHECK( cudaStreamSynchronize( stream ) );
	cudaEventElapsedTime( &traceMs, timerStart, timerStop );
	return true;
}

/*
========================
idPathTracerBackendCUDA::ReadPixels
========================
*/
bool idPathTracerBackendCUDA::ReadPixels( unsigned char *dest, int width, int height ) {
	if ( !healthy || !dest || width != lastWidth || height != lastHeight ) {
		return false;
	}
	PT_CUDA_CHECK( cudaMemcpy( dest, d_output, (size_t)width * height * 4, cudaMemcpyDeviceToHost ) );
	return true;
}

/*
========================
idPathTracerBackendCUDA::GetStats
========================
*/
void idPathTracerBackendCUDA::GetStats( ptBackendStats_t &stats ) const {
	stats.traceMs = traceMs;
	stats.numVertices = numVertices;
	stats.numTriangles = numTriangles;
	stats.numBVHNodes = numBVHNodes;
	stats.numMaterials = numMaterials;
	stats.numLights = numLights;
	stats.numTextures = numTextures;
	stats.deviceBytes = textureBytes
		+ (size_t)maxWidth * maxHeight * 4 * ( sizeof( float ) + 1 )
		+ (size_t)capVertices * sizeof( ptVertex_t )
		+ (size_t)capTriangles * sizeof( ptTriangle_t )
		+ (size_t)capBVHNodes * sizeof( ptBVHNode_t )
		+ (size_t)( capTriIndices + capSorted + capParents + capAtomics ) * sizeof( int )
		+ (size_t)( capMorton + capMortonAlt ) * sizeof( unsigned int )
		+ (size_t)capSortTemp
		+ (size_t)capMaterials * sizeof( ptMaterial_t )
		+ (size_t)capLights * sizeof( ptLight_t )
		+ PT_MAX_TEXTURES * sizeof( cudaTexture_t );
}

/*
========================
Factory
========================
*/
bool PT_CUDA_IsAvailable() {
	return idPathTracerBackendCUDA::IsAvailable();
}

idPathTracerBackend *PT_CUDA_CreateBackend() {
	return new idPathTracerBackendCUDA();
}
