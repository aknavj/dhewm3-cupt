#include "renderer_pt/cuda/pt_cuda_postprocess.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static void Check( bool condition, const char *label ) {
	if ( !condition ) {
		std::fprintf( stderr, "FAIL: %s\n", label );
		std::exit( 1 );
	}
}

static void Cuda( cudaError_t error ) {
	if ( error != cudaSuccess ) {
		std::fprintf( stderr, "CUDA: %s\n", cudaGetErrorString( error ) );
		std::exit( 1 );
	}
}

struct fixture_t {
	int width, height;
	float *input, *history;
	ptCudaPostprocessBuffers_t buffers;
	std::vector<float> source, originalHistory;

	fixture_t( int w, int h ) : width( w ), height( h ), buffers(),
		source( w * h * 4, 0 ), originalHistory( w * h * 4, 0.375f ) {
		const size_t bytes = source.size() * sizeof( float );
		Cuda( cudaMalloc( &input, bytes ) );
		Cuda( cudaMalloc( &history, bytes ) );
		Cuda( cudaMalloc( &buffers.output, bytes ) );
		// Maximum scratch size for every supported radius (minimum downsample is four).
		const size_t smallBytes = (size_t)( ( w + 3 ) / 4 ) * ( ( h + 3 ) / 4 ) * sizeof( float4 );
		Cuda( cudaMalloc( &buffers.highlights, smallBytes ) );
		Cuda( cudaMalloc( &buffers.scratch, smallBytes ) );
		Cuda( cudaMalloc( &buffers.blurred, smallBytes ) );
		Cuda( cudaMemcpy( history, originalHistory.data(), bytes, cudaMemcpyHostToDevice ) );
		for ( size_t i = 3; i < source.size(); i += 4 ) source[i] = 0.75f;
	}

	~fixture_t() {
		Cuda( cudaFree( input ) ); Cuda( cudaFree( history ) );
		Cuda( cudaFree( buffers.output ) ); Cuda( cudaFree( buffers.highlights ) );
		Cuda( cudaFree( buffers.scratch ) ); Cuda( cudaFree( buffers.blurred ) );
	}

	std::vector<float> Run( float bloom, float flare, float exposure = 1, float threshold = 1, float radius = 0.025f ) {
		const size_t bytes = source.size() * sizeof( float );
		Cuda( cudaMemcpy( input, source.data(), bytes, cudaMemcpyHostToDevice ) );
		Cuda( PT_Postprocess( buffers, input, width, height, exposure, threshold, radius, bloom, flare, 0 ) );
		Cuda( cudaDeviceSynchronize() );
		std::vector<float> result( source.size() ), unchanged( source.size() );
		Cuda( cudaMemcpy( result.data(), buffers.output, bytes, cudaMemcpyDeviceToHost ) );
		Cuda( cudaMemcpy( unchanged.data(), input, bytes, cudaMemcpyDeviceToHost ) );
		Check( std::memcmp( unchanged.data(), source.data(), bytes ) == 0, "HDR input untouched" );
		Cuda( cudaMemcpy( unchanged.data(), history, bytes, cudaMemcpyDeviceToHost ) );
		Check( unchanged == originalHistory, "history untouched" );
		for ( float value : result ) Check( std::isfinite( value ), "finite output" );
		for ( size_t i = 3; i < source.size(); i += 4 )
			Check( result[i] == source[i], "alpha preserved" );
		return result;
	}
};

static double Energy( const std::vector<float> &image ) {
	double total = 0;
	for ( size_t i = 0; i < image.size(); i++ ) if ( i % 4 != 3 ) total += image[i];
	return total;
}

int main() {
	int devices = 0;
	if ( cudaGetDeviceCount( &devices ) != cudaSuccess || !devices ) return 77;
	fixture_t f( 129, 97 );
	Check( Energy( f.Run( 1, 1 ) ) == 0, "constant black stays black" );
	const size_t bright = ( 27 * f.width + 31 ) * 4;
	f.source[bright] = f.source[bright + 1] = f.source[bright + 2] = 64;
	Check( f.Run( 0, 0 ) == f.source, "disabled identity" );
	Check( f.Run( -1, -1 ) == f.source, "negative strengths disable effects" );
	const std::vector<float> bloom = f.Run( 1, 0 );
	Check( bloom[bright + 4] > 0, "isolated highlight spreads" );
	Check( Energy( bloom ) > Energy( f.source ), "bloom adds highlight energy" );
	const std::vector<float> flare = f.Run( 0, 1 );
	Check( Energy( flare ) > Energy( f.source ), "flare independently enabled" );
	const size_t reflected = ( ( f.height - 1 - 27 ) * f.width + f.width - 1 - 31 ) * 4;
	Check( flare[reflected] > 0, "ghost reaches reflected screen location" );
	Check( f.Run( 1, 1, 1, 100 ) == f.source, "below exposed threshold unchanged" );
	Check( f.Run( 1, 1, 1, 64 ) == f.source, "at exposed threshold unchanged" );
	Check( Energy( f.Run( 1, 0, 2, 100 ) ) > Energy( f.source ), "exposure changes threshold" );
	std::vector<float> scaledSource = f.source;
	for ( size_t i = 0; i < f.source.size(); i++ ) if ( i % 4 != 3 ) f.source[i] *= 2;
	const std::vector<float> doubled = f.Run( 1, 0, 0.5f );
	for ( size_t i = 0; i < bloom.size(); i++ ) if ( i % 4 != 3 )
		Check( std::fabs( doubled[i] * 0.5f - bloom[i] ) < 1e-5f, "exposure-relative equivalence" );
	f.source = scaledSource;
	f.source[0] = std::numeric_limits<float>::infinity();
	f.source[1] = std::numeric_limits<float>::quiet_NaN();
	f.source[2] = -1;
	f.Run( 4, 1, 1, 0 );
	f.Run( 1, 1, 1, 1, std::numeric_limits<float>::infinity() );
	f.Run( 1, 1, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity() );
	f.source[0] = f.source[1] = f.source[2] = std::numeric_limits<float>::max();
	f.Run( 4, 1, std::numeric_limits<float>::max(), 0, 0.1f );
	fixture_t tiny( 1, 1 );
	tiny.source[0] = tiny.source[1] = tiny.source[2] = 16;
	tiny.Run( 1, 1 );
	fixture_t adaptive( 43, 513 );
	adaptive.source[0] = adaptive.source[1] = adaptive.source[2] = 64;
	adaptive.Run( 1, 1, 1, 1, 0.1f );
	Check( PT_CudaPostprocessScale( 1080, 0.025f ) * 2 == PT_CudaPostprocessScale( 2160, 0.025f ),
		"radius scales with resolution" );
	std::puts( "CUDA postprocess tests passed" );
	return 0;
}
