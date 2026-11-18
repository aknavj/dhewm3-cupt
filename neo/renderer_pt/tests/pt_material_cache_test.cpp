#include "sys/platform.h"
#include "renderer_pt/pt_material_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK( condition ) do { \
	if ( !( condition ) ) { \
		fprintf( stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition ); \
		exit( 1 ); \
	} \
} while ( 0 )

struct cachedValue_t {
	const void *material;
	unsigned int nameHash;
	unsigned int regHash;
	bool animated;
};

static int Acquire( idPathTracerMaterialCache &cache, std::vector<cachedValue_t> &values,
					const void *material, unsigned int nameHash, bool animated,
					unsigned int regHash, int frame, bool &rebuild ) {
	const int index = cache.Acquire( material, nameHash, animated, regHash, frame, rebuild );
	CHECK( index > PT_DEFAULT_MATERIAL );
	if ( index >= (int)values.size() ) {
		values.resize( index + 1 );
	}
	if ( rebuild ) {
		const cachedValue_t value = { material, nameHash, regHash, animated };
		values[index] = value;
	} else {
		CHECK( values[index].material == material );
		CHECK( values[index].nameHash == nameHash );
		CHECK( values[index].animated == animated );
		CHECK( !animated || values[index].regHash == regHash );
	}
	return index;
}

int main() {
	idPathTracerMaterialCache cache;
	std::vector<cachedValue_t> values( 1 );
	std::vector<int> materials( 6000 );
	bool rebuild;

	cache.BeginFrame();
	for ( int i = 0; i < PT_MATERIAL_CACHE_REUSE_THRESHOLD - 1; i++ ) {
		const int index = Acquire( cache, values, &materials[i], i + 1, false, 0, 1, rebuild );
		CHECK( rebuild && index == i + 1 );
	}
	CHECK( cache.Num() == PT_MATERIAL_CACHE_REUSE_THRESHOLD );

	cache.BeginFrame();
	const int protectedIndex = Acquire( cache, values, &materials[0], 1, false, 0, 2, rebuild );
	CHECK( !rebuild && protectedIndex == 1 );
	for ( int i = 1023; i < 2045; i++ ) {
		const int index = Acquire( cache, values, &materials[i], i + 1, false, 0, 2, rebuild );
		CHECK( rebuild && index != protectedIndex && index < PT_MATERIAL_CACHE_REUSE_THRESHOLD );
	}
	CHECK( cache.Num() == PT_MATERIAL_CACHE_REUSE_THRESHOLD );
	CHECK( Acquire( cache, values, &materials[0], 1, false, 0, 2, rebuild ) == protectedIndex );
	CHECK( !rebuild );

	cache.BeginFrame();
	std::vector<int> liveIndices( 1500 );
	std::vector<bool> used( 1, false );
	for ( int i = 0; i < 1500; i++ ) {
		const int id = 2045 + i;
		const int index = Acquire( cache, values, &materials[id], id + 1, false, 0, 3, rebuild );
		if ( index >= (int)used.size() ) {
			used.resize( index + 1, false );
		}
		CHECK( !used[index] );
		used[index] = true;
		liveIndices[i] = index;
	}
	CHECK( cache.Num() == 1501 );
	for ( int i = 0; i < 1500; i++ ) {
		const int id = 2045 + i;
		CHECK( Acquire( cache, values, &materials[id], id + 1, false, 0, 3, rebuild ) == liveIndices[i] );
		CHECK( !rebuild );
	}

	cache.BeginFrame();
	const void *shared = &materials[0];
	const int constant = Acquire( cache, values, shared, 1, false, 0, 4, rebuild );
	const int animatedA = Acquire( cache, values, shared, 1, true, 11, 4, rebuild );
	const int animatedB = Acquire( cache, values, shared, 1, true, 22, 4, rebuild );
	CHECK( constant != animatedA && constant != animatedB && animatedA != animatedB );
	CHECK( Acquire( cache, values, shared, 1, false, 0, 4, rebuild ) == constant && !rebuild );
	CHECK( Acquire( cache, values, shared, 1, true, 11, 4, rebuild ) == animatedA && !rebuild );
	CHECK( Acquire( cache, values, shared, 1, true, 22, 4, rebuild ) == animatedB && !rebuild );

	cache.BeginFrame();
	CHECK( Acquire( cache, values, shared, 1, true, 11, 5, rebuild ) == animatedA && !rebuild );
	const int animatedC = Acquire( cache, values, shared, 1, true, 33, 5, rebuild );
	const int animatedBAgain = Acquire( cache, values, shared, 1, true, 22, 5, rebuild );
	CHECK( animatedA != animatedC && animatedA != animatedBAgain && animatedC != animatedBAgain );
	CHECK( Acquire( cache, values, shared, 1, true, 11, 5, rebuild ) == animatedA && !rebuild );

	cache.BeginFrame();
	const int oldName = Acquire( cache, values, shared, 1, false, 0, 6, rebuild );
	const int newName = Acquire( cache, values, shared, 99999, false, 0, 6, rebuild );
	CHECK( oldName != newName );
	CHECK( Acquire( cache, values, shared, 1, false, 0, 6, rebuild ) == oldName && !rebuild );

	for ( int frame = 7; frame < 100; frame++ ) {
		cache.BeginFrame();
		std::vector<bool> frameUsed( cache.Num(), false );
		for ( int i = 0; i < 100; i++ ) {
			const int id = ( frame * 101 + i ) % (int)materials.size();
			const int index = Acquire( cache, values, &materials[id], id + 1, false, 0, frame, rebuild );
			CHECK( index < (int)frameUsed.size() && !frameUsed[index] );
			frameUsed[index] = true;
		}
		CHECK( cache.Num() == 1501 );
	}

	cache.Clear();
	values.clear();
	values.resize( 1 );
	CHECK( cache.Num() == 1 );
	cache.BeginFrame();
	CHECK( Acquire( cache, values, shared, 1, false, 0, 100, rebuild ) == 1 && rebuild );
	printf( "material cache regression tests passed\n" );
	return 0;
}
