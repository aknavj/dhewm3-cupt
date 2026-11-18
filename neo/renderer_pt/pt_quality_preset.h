#ifndef __PT_QUALITY_PRESET_H__
#define __PT_QUALITY_PRESET_H__

#include "idlib/Str.h"

struct ptQualitySettings_t {
	float renderScale;
	int samplesPerPixel;
	int maxDepth;
	int maxLightSamples;
};

struct ptQualityPreset_t {
	const char *name;
	ptQualitySettings_t settings;
};

static const ptQualityPreset_t ptQualityPresets[] = {
	{ "low",   { 0.35f, 1, 2, 1 } },
	{ "med",   { 0.50f, 4, 3, 2 } },
	{ "high",  { 0.75f, 6, 4, 3 } },
	{ "ultra", { 1.00f, 8, 5, 4 } }
};

static const int PT_QUALITY_PRESET_COUNT = sizeof( ptQualityPresets ) / sizeof( ptQualityPresets[0] );

static ID_INLINE const ptQualityPreset_t *PT_FindQualityPreset( const char *name ) {
	if ( !name ) {
		return NULL;
	}
	for ( int i = 0; i < PT_QUALITY_PRESET_COUNT; i++ ) {
		if ( idStr::Icmp( name, ptQualityPresets[i].name ) == 0 ) {
			return &ptQualityPresets[i];
		}
	}
	return NULL;
}

static ID_INLINE bool PT_QualitySettingsEqual( const ptQualitySettings_t &a, const ptQualitySettings_t &b ) {
	return a.renderScale == b.renderScale && a.samplesPerPixel == b.samplesPerPixel
		&& a.maxDepth == b.maxDepth && a.maxLightSamples == b.maxLightSamples;
}

#endif /* !__PT_QUALITY_PRESET_H__ */
