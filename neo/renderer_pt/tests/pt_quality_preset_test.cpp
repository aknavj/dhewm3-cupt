#include "sys/platform.h"
#include "renderer_pt/pt_quality_preset.h"

#include <stdio.h>
#include <stdlib.h>

class idCommon;
idCommon *common = NULL; // idlib links against the engine global even for string helpers

#define CHECK( condition ) do { \
	if ( !( condition ) ) { \
		fprintf( stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition ); \
		exit( 1 ); \
	} \
} while ( 0 )

int main() {
	const ptQualitySettings_t expected[] = {
		{ 0.35f, 1, 2, 1 },
		{ 0.50f, 4, 3, 2 },
		{ 0.75f, 6, 4, 3 },
		{ 1.00f, 8, 5, 4 }
	};
	const char *names[] = { "low", "med", "high", "ultra" };
	CHECK( PT_QUALITY_PRESET_COUNT == 4 );
	for ( int i = 0; i < PT_QUALITY_PRESET_COUNT; i++ ) {
		const ptQualityPreset_t *preset = PT_FindQualityPreset( names[i] );
		CHECK( preset != NULL );
		CHECK( PT_QualitySettingsEqual( preset->settings, expected[i] ) );
		if ( i > 0 ) {
			const ptQualitySettings_t &previous = expected[i - 1];
			CHECK( preset->settings.renderScale > previous.renderScale );
			CHECK( preset->settings.samplesPerPixel > previous.samplesPerPixel );
			CHECK( preset->settings.maxDepth > previous.maxDepth );
			CHECK( preset->settings.maxLightSamples > previous.maxLightSamples );
		}
	}
	CHECK( PT_FindQualityPreset( "HIGH" ) == &ptQualityPresets[2] );
	CHECK( PT_FindQualityPreset( "custom" ) == NULL );
	CHECK( PT_FindQualityPreset( "invalid" ) == NULL );
	CHECK( PT_FindQualityPreset( NULL ) == NULL );

	ptQualitySettings_t edited = expected[1];
	edited.renderScale = 0.6f;
	CHECK( !PT_QualitySettingsEqual( edited, expected[1] ) );
	edited = expected[1];
	edited.samplesPerPixel++;
	CHECK( !PT_QualitySettingsEqual( edited, expected[1] ) );
	edited = expected[1];
	edited.maxDepth++;
	CHECK( !PT_QualitySettingsEqual( edited, expected[1] ) );
	edited = expected[1];
	edited.maxLightSamples++;
	CHECK( !PT_QualitySettingsEqual( edited, expected[1] ) );

	puts( "quality preset regression tests passed" );
	return 0;
}
