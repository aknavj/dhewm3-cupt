#ifndef __PT_PHYSICAL_H__
#define __PT_PHYSICAL_H__

#ifdef PT_PHYSICAL

#include <math.h>
#include "renderer_pt/pt_types.h"

#ifdef __CUDACC__
#define PT_PHYSICAL_INLINE __host__ __device__ inline
#else
#define PT_PHYSICAL_INLINE inline
#endif

PT_PHYSICAL_INLINE bool PT_PhysicalActive( const ptRenderParams_t *params ) {
	return params && params->physicalIor > 0.0f;
}

PT_PHYSICAL_INLINE float PT_PhysicalEV100( float fNumber, float shutterSeconds, float iso ) {
	return log2f( fNumber * fNumber * 100.0f / ( shutterSeconds * iso ) );
}

PT_PHYSICAL_INLINE float PT_PhysicalExposure( float fNumber, float shutterSeconds, float iso, float compensation ) {
	return exp2f( compensation - PT_PhysicalEV100( fNumber, shutterSeconds, iso ) ) / 1.2f;
}

PT_PHYSICAL_INLINE float PT_PhysicalFov( float sensorMillimeters, float focalMillimeters ) {
	return 2.0f * atanf( sensorMillimeters / ( 2.0f * focalMillimeters ) ) * ( 180.0f / 3.14159265358979f );
}

PT_PHYSICAL_INLINE float PT_PhysicalApertureRadius( float focalMillimeters, float fNumber, float metersPerUnit ) {
	return focalMillimeters * 0.001f / ( 2.0f * fNumber * metersPerUnit );
}

PT_PHYSICAL_INLINE bool PT_PhysicalTraceChanged( const ptRenderParams_t &a, const ptRenderParams_t &b ) {
	return a.physicalMetersPerUnit != b.physicalMetersPerUnit
		|| a.physicalLightLumens != b.physicalLightLumens || a.physicalSunLux != b.physicalSunLux
		|| a.physicalIor != b.physicalIor || a.physicalRoughnessScale != b.physicalRoughnessScale
		|| a.physicalApertureRadius != b.physicalApertureRadius
		|| a.physicalFogDensity != b.physicalFogDensity || a.physicalFogAlbedo != b.physicalFogAlbedo
		|| a.physicalFogAnisotropy != b.physicalFogAnisotropy
		|| a.physicalFogRadius != b.physicalFogRadius || a.physicalFogSteps != b.physicalFogSteps
		|| a.physicalFocusDistance != b.physicalFocusDistance;
}

#undef PT_PHYSICAL_INLINE

#ifndef __CUDACC__
struct viewDef_s;
void R_PT_InitPhysical();
void R_PT_PhysicalCamera( struct viewDef_s *view );
void R_PT_PhysicalParams( ptRenderParams_t &params );
void R_PT_DrawPhysicalPanel();
#endif

#endif
#endif
