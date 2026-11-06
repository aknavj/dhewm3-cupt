#include "sys/platform.h"

#ifdef HAVE_PATHTRACER

#include "framework/Common.h"
#include "framework/CmdSystem.h"
#include "renderer/tr_local.h"
#include "renderer_pt/pt_quality_preset.h"
#include "renderer_pt/pt_render.h"
#include "renderer_pt/pt_scene.h"

/*
===============================================================================

	Path tracer front end

	Owns the backend and the scene builder, translates cvars into
	ptRenderParams_t and composites the result into the view's viewport.
	Everything here is compute-API agnostic.

===============================================================================
*/

idCVar r_ptDraw( "r_ptDraw", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_BOOL, "draw 3D views with the path tracer instead of OpenGL" );
idCVar r_ptBackend( "r_ptBackend", "auto", CVAR_RENDERER | CVAR_ARCHIVE, "path tracer compute backend: auto, cuda" );
idCVar r_ptMode( "r_ptMode", "0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "0 = path tracing, 1 = BVH debug, 2 = textured combined, 3 = base color, 4 = albedo texture, 5 = normal map, 6 = specular map, 7 = Whitted ray tracing", 0, PT_MODE_COUNT - 1 );
idCVar r_ptDebug( "r_ptDebug", "0", CVAR_RENDERER | CVAR_BOOL, "print path tracer scene / timing info" );

// quality
static void PT_PresetCompletion( const idCmdArgs &args, void ( *callback )( const char * ) ) {
	for ( int i = 0; i < PT_QUALITY_PRESET_COUNT; i++ ) {
		callback( va( "%s %s", args.Argv( 0 ), ptQualityPresets[i].name ) );
	}
	callback( va( "%s custom", args.Argv( 0 ) ) );
}

idCVar r_ptPreset( "r_ptPreset", "med", CVAR_RENDERER | CVAR_ARCHIVE,
	"path tracer quality preset: low, med, high, ultra, or custom", PT_PresetCompletion );
idCVar r_ptRenderScale( "r_ptRenderScale", "0.5", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "trace resolution relative to the viewport", 0.1f, 1.0f );
idCVar r_ptSamplesPerPixel( "r_ptSamplesPerPixel", "4", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "samples per pixel per frame", 1, 64 );
idCVar r_ptMaxDepth( "r_ptMaxDepth", "3", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "maximum ray bounces", 1, 16 );
idCVar r_ptMaxLightSamples( "r_ptMaxLightSamples", "2", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "lights sampled per hit, picked by estimated contribution", 1, 8 );
idCVar r_ptIndirectProb( "r_ptIndirectProb", "0.15", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "probability of the one bounce GI estimate, only used when r_ptMaxDepth is 1", 0.0f, 1.0f );

// path termination
idCVar r_ptRussianRoulette( "r_ptRussianRoulette", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_BOOL, "Russian roulette path termination" );
idCVar r_ptRRMinBounces( "r_ptRRMinBounces", "2", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "bounces before Russian roulette starts", 0, 16 );
idCVar r_ptRRSurvivalMin( "r_ptRRSurvivalMin", "0.1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "minimum Russian roulette survival probability", 0.01f, 0.95f );
idCVar r_ptEarlyTermThreshold( "r_ptEarlyTermThreshold", "0.05", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "Russian roulette for paths below this throughput after a bounce (unbiased)", 0.0f, 0.5f );
idCVar r_ptFireflyClamp( "r_ptFireflyClamp", "5.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "clamp per-sample radiance (0 = off)" );
idCVar r_ptThroughputClamp( "r_ptThroughputClamp", "0.25", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "Russian roulette for paths below this throughput after shading (unbiased)", 0.0f, 1.0f );

// accumulation
idCVar r_ptAccumulation( "r_ptAccumulation", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_BOOL, "accumulate samples over frames while the view is static" );
idCVar r_ptResetOnSceneChange( "r_ptResetOnSceneChange", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "restart accumulation on scene changes: 0 = camera only, 1 = objects / lights appearing or vanishing, 2 = any change incl. animation (may never converge)", 0, 2 );

// shading
idCVar r_ptRayOffset( "r_ptRayOffset", "0.001", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "ray origin offset against self intersection", 0.0001f, 0.1f );
idCVar r_ptEmissionBoost( "r_ptEmissionBoost", "1.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "emission multiplier" );
idCVar r_ptSpecularBoost( "r_ptSpecularBoost", "2.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "specular BRDF multiplier (Doom 3 uses 2x)" );
idCVar r_ptSoftShadowScale( "r_ptSoftShadowScale", "0.2", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "soft shadow jitter (0 = hard shadows)" );

// display
idCVar r_ptExposure( "r_ptExposure", "1.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "exposure multiplier" );
idCVar r_ptGamma( "r_ptGamma", "2.2", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "output gamma, textures and colors are decoded with 2.2", 0.1f, 10.0f );
idCVar r_ptToneMapMode( "r_ptToneMapMode", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "0 = Reinhard, 1 = ACES, 2 = Uncharted 2", 0, 2 );

// sky
idCVar r_ptSkyIntensity( "r_ptSkyIntensity", "0.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "sky dome intensity for primary ray misses" );
idCVar r_ptSkyColorZenithR( "r_ptSkyColorZenithR", "0.3", CVAR_RENDERER | CVAR_FLOAT, "sky zenith red" );
idCVar r_ptSkyColorZenithG( "r_ptSkyColorZenithG", "0.35", CVAR_RENDERER | CVAR_FLOAT, "sky zenith green" );
idCVar r_ptSkyColorZenithB( "r_ptSkyColorZenithB", "0.5", CVAR_RENDERER | CVAR_FLOAT, "sky zenith blue" );
idCVar r_ptSkyColorHorizonR( "r_ptSkyColorHorizonR", "0.4", CVAR_RENDERER | CVAR_FLOAT, "sky horizon red" );
idCVar r_ptSkyColorHorizonG( "r_ptSkyColorHorizonG", "0.35", CVAR_RENDERER | CVAR_FLOAT, "sky horizon green" );
idCVar r_ptSkyColorHorizonB( "r_ptSkyColorHorizonB", "0.3", CVAR_RENDERER | CVAR_FLOAT, "sky horizon blue" );
idCVar r_ptSkyColorGroundR( "r_ptSkyColorGroundR", "0.1", CVAR_RENDERER | CVAR_FLOAT, "sky ground red" );
idCVar r_ptSkyColorGroundG( "r_ptSkyColorGroundG", "0.08", CVAR_RENDERER | CVAR_FLOAT, "sky ground green" );
idCVar r_ptSkyColorGroundB( "r_ptSkyColorGroundB", "0.05", CVAR_RENDERER | CVAR_FLOAT, "sky ground blue" );

// volumetrics
idCVar r_ptVolumetric( "r_ptVolumetric", "0.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "volumetric light scattering intensity (0 = off, expensive)" );
idCVar r_ptVolumetricDensity( "r_ptVolumetricDensity", "0.008", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "volumetric fog density" );
idCVar r_ptVolumetricSteps( "r_ptVolumetricSteps", "16", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_INTEGER, "volumetric ray march steps", 1, 128 );
idCVar r_ptVolumetricAnisotropy( "r_ptVolumetricAnisotropy", "0.6", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "Henyey-Greenstein anisotropy", -0.999f, 0.999f );
idCVar r_ptVolFalloff( "r_ptVolFalloff", "2.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "volumetric distance falloff exponent" );
idCVar r_ptVolMaxDist( "r_ptVolMaxDist", "500.0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT, "volumetric max sampling distance" );

extern void R_PT_Compare_f( const idCmdArgs &args );
extern void R_PT_Sequence_f( const idCmdArgs &args );
extern void R_PT_Abort_f( const idCmdArgs &args );

/*
================
Backend registry

To add a backend, implement idPathTracerBackend and append its factory here.
================
*/
static const ptBackendFactory_t ptBackendFactories[] = {
#ifdef HAVE_CUDA
	{ "cuda", PT_CUDA_IsAvailable, PT_CUDA_CreateBackend },
#endif
	{ NULL, NULL, NULL }
};

typedef struct {
	idPathTracerBackend *	backend;
	idPathTracerScene *		scene;
	bool					initFailed;		// latched until r_ptBackend changes or vid_restart

	// last traced view, for accumulation invalidation
	ptRenderParams_t		prevParams;
	unsigned int			prevContentHash;
	unsigned int			prevStructureHash;
	float					prevLightScale;
	bool					havePrev;

	// display
	GLuint					glTexture;
	int						texWidth;
	int						texHeight;
	idList<byte>			pixels;
} ptFrontEnd_t;

static ptFrontEnd_t pt;

static bool ptApplyingPreset = false;
static const char *ptLastPresetName = "med";

static void PT_PresetChanged( const idCVar &cvar ) {
	if ( ptApplyingPreset ) {
		return;
	}
	if ( idStr::Icmp( cvar.GetString(), "custom" ) == 0 ) {
		ptLastPresetName = "custom";
		return;
	}
	const ptQualityPreset_t *preset = PT_FindQualityPreset( cvar.GetString() );
	if ( !preset ) {
		common->Warning( "invalid r_ptPreset '%s'; use low, med, high, ultra, or custom", cvar.GetString() );
		ptApplyingPreset = true;
		r_ptPreset.SetString( ptLastPresetName );
		ptApplyingPreset = false;
		return;
	}

	ptLastPresetName = preset->name;
	ptApplyingPreset = true;
	r_ptRenderScale.SetFloat( preset->settings.renderScale );
	r_ptSamplesPerPixel.SetInteger( preset->settings.samplesPerPixel );
	r_ptMaxDepth.SetInteger( preset->settings.maxDepth );
	r_ptMaxLightSamples.SetInteger( preset->settings.maxLightSamples );
	ptApplyingPreset = false;
}

static void PT_QualityChanged( const idCVar & ) {
	if ( ptApplyingPreset || idStr::Icmp( r_ptPreset.GetString(), "custom" ) == 0 ) {
		return;
	}
	const ptQualityPreset_t *preset = PT_FindQualityPreset( r_ptPreset.GetString() );
	if ( !preset ) {
		common->Warning( "cannot update quality for invalid r_ptPreset '%s'", r_ptPreset.GetString() );
		return;
	}
	const ptQualitySettings_t current = {
		r_ptRenderScale.GetFloat(), r_ptSamplesPerPixel.GetInteger(),
		r_ptMaxDepth.GetInteger(), r_ptMaxLightSamples.GetInteger()
	};
	if ( !PT_QualitySettingsEqual( current, preset->settings ) ) {
		r_ptPreset.SetString( "custom" );
	}
}

static void PT_InitQualityPresets() {
	idCVar *qualityCVars[] = { &r_ptRenderScale, &r_ptSamplesPerPixel, &r_ptMaxDepth, &r_ptMaxLightSamples };
	if ( !CVar_RegisterChangeCallback( r_ptPreset, PT_PresetChanged ) ) {
		common->FatalError( "could not register r_ptPreset callback" );
	}
	for ( int i = 0; i < (int)( sizeof( qualityCVars ) / sizeof( qualityCVars[0] ) ); i++ ) {
		if ( !CVar_RegisterChangeCallback( *qualityCVars[i], PT_QualityChanged ) ) {
			common->FatalError( "could not register %s quality callback", qualityCVars[i]->GetName() );
		}
	}
	PT_PresetChanged( r_ptPreset );
}

void RB_PT_InvalidateImages( void ) {
	if ( pt.backend ) {
		pt.backend->ResetAccumulation();
	}
	if ( pt.scene ) {
		pt.scene->Clear();
	}
	pt.havePrev = false;
}

/*
========================
PT_FindFactory
========================
*/
static const ptBackendFactory_t *PT_FindFactory( const char *name ) {
	const bool any = ( name[0] == '\0' || idStr::Icmp( name, "auto" ) == 0 );
	for ( int i = 0; ptBackendFactories[i].name; i++ ) {
		const ptBackendFactory_t &f = ptBackendFactories[i];
		if ( any ? f.isAvailable() : idStr::Icmp( name, f.name ) == 0 ) {
			return &f;
		}
	}
	return NULL;
}

/*
========================
PT_DestroyBackend
========================
*/
static void PT_DestroyBackend() {
	if ( pt.backend ) {
		pt.backend->Shutdown();
		delete pt.backend;
		pt.backend = NULL;
	}
	if ( pt.scene ) {
		delete pt.scene;
		pt.scene = NULL;
	}
	pt.havePrev = false;
}

/*
========================
PT_EnsureBackend
========================
*/
static bool PT_EnsureBackend() {
	if ( r_ptBackend.IsModified() ) {
		r_ptBackend.ClearModified();
		PT_DestroyBackend();
		pt.initFailed = false;
	}

	if ( pt.backend ) {
		if ( pt.backend->IsHealthy() ) {
			return true;
		}
		common->Warning( "Path tracer: backend '%s' failed, falling back to OpenGL (vid_restart or change r_ptBackend to retry)\n", pt.backend->Name() );
		PT_DestroyBackend();
		pt.initFailed = true;
		return false;
	}

	if ( pt.initFailed ) {
		return false;
	}

	const ptBackendFactory_t *factory = PT_FindFactory( r_ptBackend.GetString() );
	if ( !factory || !factory->isAvailable() ) {
		common->Warning( "Path tracer: backend '%s' is not available, using OpenGL\n", r_ptBackend.GetString() );
		pt.initFailed = true;
		return false;
	}

	pt.backend = factory->create();
	if ( !pt.backend || !pt.backend->Init( glConfig.vidWidth, glConfig.vidHeight ) ) {
		common->Warning( "Path tracer: backend '%s' failed to initialize, using OpenGL\n", factory->name );
		PT_DestroyBackend();
		pt.initFailed = true;
		return false;
	}

	pt.scene = new idPathTracerScene();
	common->Printf( "Path tracer active (backend '%s'), 'r_ptDraw 0' switches back to OpenGL\n", pt.backend->Name() );
	return true;
}

/*
========================
RB_PT_GetBackend
========================
*/
idPathTracerBackend *RB_PT_GetBackend( void ) {
	return pt.backend;
}

/*
========================
RB_PT_IsSupported
========================
*/
bool RB_PT_IsSupported( void ) {
	if ( pt.backend ) {
		return pt.backend->IsHealthy();
	}
	const ptBackendFactory_t *factory = PT_FindFactory( r_ptBackend.GetString() );
	return factory && factory->isAvailable();
}

/*
========================
RB_PT_IsActive
========================
*/
bool RB_PT_IsActive( void ) {
	return r_ptDraw.GetBool() && pt.backend && pt.backend->IsHealthy();
}

/*
========================
PT_Info_f
========================
*/
static void PT_Info_f( const idCmdArgs &args ) {
	common->Printf( "compiled backends:" );
	for ( int i = 0; ptBackendFactories[i].name; i++ ) {
		common->Printf( " %s%s", ptBackendFactories[i].name, ptBackendFactories[i].isAvailable() ? "" : " (unavailable)" );
	}
	common->Printf( "\n" );
	if ( !pt.backend ) {
		common->Printf( "no active backend%s\n", pt.initFailed ? " (initialization failed)" : "" );
		return;
	}
	common->Printf( "active backend: %s%s\n", pt.backend->Name(), pt.backend->IsHealthy() ? "" : " (FAILED)" );
	pt.backend->PrintInfo();
}

/*
========================
PT_Reset_f
========================
*/
static void PT_Reset_f( const idCmdArgs &args ) {
	if ( pt.backend ) {
		pt.backend->ResetAccumulation();
	}
	if ( pt.scene ) {
		pt.scene->Clear();
	}
}

/*
========================
R_PT_InitCommands

Called once from R_InitCommands, not on every vid_restart.
========================
*/
void R_PT_InitCommands( void ) {
	PT_InitQualityPresets();
	cmdSystem->AddCommand( "r_ptInfo", PT_Info_f, CMD_FL_RENDERER, "prints path tracer backend information" );
	cmdSystem->AddCommand( "r_ptReset", PT_Reset_f, CMD_FL_RENDERER, "restarts accumulation and drops cached materials / textures" );
	cmdSystem->AddCommand( "r_ptCompare", R_PT_Compare_f, CMD_FL_RENDERER, "saves an OpenGL and a converged path traced screenshot of the current view" );
	cmdSystem->AddCommand( "r_ptSequence", R_PT_Sequence_f, CMD_FL_RENDERER, "path traces a range of demo frames to screenshots" );
	cmdSystem->AddCommand( "r_ptAbort", R_PT_Abort_f, CMD_FL_RENDERER, "aborts r_ptCompare / r_ptSequence" );
}

/*
========================
RB_PT_Init

Called whenever the GL context is (re)created. The backend itself is created
lazily, so toggling r_ptDraw at runtime works. A full vid_restart does not go
through RB_PT_Shutdown: the compute backend survives (it does not depend on the
GL context), but the blit texture died with the old context and the scene's
texture cache must be re-read from the reloaded images.
========================
*/
void RB_PT_Init( void ) {
	pt.glTexture = 0;
	pt.texWidth = pt.texHeight = 0;
	pt.initFailed = false;
	if ( pt.scene ) {
		pt.scene->Clear();
	}
	if ( pt.backend ) {
		pt.backend->ResetAccumulation();
	}
}

/*
========================
RB_PT_Shutdown
========================
*/
void RB_PT_Shutdown( void ) {
	PT_DestroyBackend();
	if ( pt.glTexture ) {
		qglDeleteTextures( 1, &pt.glTexture );
		pt.glTexture = 0;
	}
	pt.texWidth = pt.texHeight = 0;
	pt.pixels.Clear();
}

/*
========================
PT_SetupParams
========================
*/
static void PT_SetupParams( ptRenderParams_t &p, const viewDef_t *viewDef, int traceWidth, int traceHeight ) {
	memset( &p, 0, sizeof( p ) );

	const renderView_t &rv = viewDef->renderView;
	const idMat3 &axis = rv.viewaxis;
	for ( int i = 0; i < 3; i++ ) {
		p.camPos[i] = rv.vieworg[i];
		p.camForward[i] = axis[0][i];
		p.camRight[i] = -axis[1][i];
		p.camUp[i] = axis[2][i];
	}
	p.fovX = rv.fov_x;
	p.fovY = rv.fov_y;
	p.width = traceWidth;
	p.height = traceHeight;
	p.renderMode = r_ptMode.GetInteger();

	p.samplesPerPixel = r_ptSamplesPerPixel.GetInteger();
	p.maxDepth = r_ptMaxDepth.GetInteger();
	p.maxLightSamples = r_ptMaxLightSamples.GetInteger();
	p.indirectProb = r_ptIndirectProb.GetFloat();

	p.rrEnabled = r_ptRussianRoulette.GetBool() ? 1 : 0;
	p.rrMinBounces = r_ptRRMinBounces.GetInteger();
	p.rrSurvivalMin = r_ptRRSurvivalMin.GetFloat();
	p.earlyTermThreshold = r_ptEarlyTermThreshold.GetFloat();
	p.throughputClamp = r_ptThroughputClamp.GetFloat();
	p.fireflyClamp = Max( 0.0f, r_ptFireflyClamp.GetFloat() );

	p.rayOffset = r_ptRayOffset.GetFloat();
	p.emissionBoost = Max( 0.0f, r_ptEmissionBoost.GetFloat() );
	p.specularBoost = Max( 0.0f, r_ptSpecularBoost.GetFloat() );
	p.softShadowScale = Max( 0.0f, r_ptSoftShadowScale.GetFloat() );

	p.skyIntensity = Max( 0.0f, r_ptSkyIntensity.GetFloat() );
	p.skyZenith[0] = r_ptSkyColorZenithR.GetFloat();
	p.skyZenith[1] = r_ptSkyColorZenithG.GetFloat();
	p.skyZenith[2] = r_ptSkyColorZenithB.GetFloat();
	p.skyHorizon[0] = r_ptSkyColorHorizonR.GetFloat();
	p.skyHorizon[1] = r_ptSkyColorHorizonG.GetFloat();
	p.skyHorizon[2] = r_ptSkyColorHorizonB.GetFloat();
	p.skyGround[0] = r_ptSkyColorGroundR.GetFloat();
	p.skyGround[1] = r_ptSkyColorGroundG.GetFloat();
	p.skyGround[2] = r_ptSkyColorGroundB.GetFloat();

	p.volDensity = Max( 0.0f, r_ptVolumetricDensity.GetFloat() );
	p.volSteps = r_ptVolumetricSteps.GetInteger();
	p.volAnisotropy = r_ptVolumetricAnisotropy.GetFloat();
	p.volFalloff = Max( 0.0f, r_ptVolFalloff.GetFloat() );
	p.volMaxDist = Max( 0.0f, r_ptVolMaxDist.GetFloat() );

	p.exposure = Max( 0.001f, r_ptExposure.GetFloat() );
	p.gamma = r_ptGamma.GetFloat();
	p.toneMapMode = r_ptToneMapMode.GetInteger();
}

/*
========================
PT_ApplyProgressiveQuality

Cheap first frames after a reset for responsiveness, ramping up to the
cvar targets while the view stays static.
========================
*/
static void PT_ApplyProgressiveQuality( ptRenderParams_t &p, int frame ) {
	if ( frame < 2 ) {
		p.samplesPerPixel = 1;
		p.maxDepth = Min( p.maxDepth, 2 );
		p.maxLightSamples = 1;
		p.indirectProb = 0.0f;
	} else if ( frame < 4 ) {
		p.samplesPerPixel = Min( p.samplesPerPixel, 2 );
		p.maxDepth = Min( p.maxDepth, 2 );
		p.maxLightSamples = 1;
		p.indirectProb *= 0.25f;
	} else if ( frame < 8 ) {
		p.maxLightSamples = Min( p.maxLightSamples, p.maxLightSamples / 2 + 1 );
		p.indirectProb *= 0.5f;
	}
}

/*
========================
PT_ViewChanged
========================
*/
static bool PT_ViewChanged( const ptRenderParams_t &a, const ptRenderParams_t &b ) {
	const float eps = 1e-4f;
	for ( int i = 0; i < 3; i++ ) {
		if ( idMath::Fabs( a.camPos[i] - b.camPos[i] ) > eps ||
			 idMath::Fabs( a.camForward[i] - b.camForward[i] ) > eps ||
			 idMath::Fabs( a.camUp[i] - b.camUp[i] ) > eps ) {
			return true;
		}
	}
	return idMath::Fabs( a.fovX - b.fovX ) > eps || idMath::Fabs( a.fovY - b.fovY ) > eps
		|| a.width != b.width || a.height != b.height || a.renderMode != b.renderMode
		|| a.samplesPerPixel != b.samplesPerPixel || a.maxDepth != b.maxDepth
		|| a.maxLightSamples != b.maxLightSamples
		|| a.emissionBoost != b.emissionBoost;
}

/*
========================
PT_BlitToViewport

Uploads the traced image into a texture and stretches it over the view's
viewport with bilinear filtering.
========================
*/
static void PT_BlitToViewport( const byte *rgba, int width, int height ) {
	// let the engine bring TMU 0 into a known 2D state, then take over the binding
	GL_SelectTexture( 0 );
	globalImages->whiteImage->Bind();

	if ( !pt.glTexture ) {
		qglGenTextures( 1, &pt.glTexture );
		pt.texWidth = pt.texHeight = 0;
	}
	qglBindTexture( GL_TEXTURE_2D, pt.glTexture );
	backEnd.glState.tmu[0].current2DMap = -1;	// force the next idImage::Bind to rebind

	if ( width != pt.texWidth || height != pt.texHeight ) {
		qglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		qglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		qglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		qglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		qglTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba );
		pt.texWidth = width;
		pt.texHeight = height;
	} else {
		qglTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba );
	}

	GL_TexEnv( GL_REPLACE );
	GL_State( GLS_DEPTHFUNC_ALWAYS );
	GL_Cull( CT_TWO_SIDED );

	qglMatrixMode( GL_PROJECTION );
	qglPushMatrix();
	qglLoadIdentity();
	qglOrtho( 0, 1, 0, 1, -1, 1 );
	qglMatrixMode( GL_MODELVIEW );
	qglPushMatrix();
	qglLoadIdentity();

	qglColor4f( 1.0f, 1.0f, 1.0f, 1.0f );
	qglBegin( GL_QUADS );
	qglTexCoord2f( 0.0f, 0.0f ); qglVertex2f( 0.0f, 0.0f );
	qglTexCoord2f( 1.0f, 0.0f ); qglVertex2f( 1.0f, 0.0f );
	qglTexCoord2f( 1.0f, 1.0f ); qglVertex2f( 1.0f, 1.0f );
	qglTexCoord2f( 0.0f, 1.0f ); qglVertex2f( 0.0f, 1.0f );
	qglEnd();

	qglMatrixMode( GL_PROJECTION );
	qglPopMatrix();
	qglMatrixMode( GL_MODELVIEW );
	qglPopMatrix();

	GL_TexEnv( GL_MODULATE );
}

/*
========================
RB_PT_DrawView

Returns false when the view should be drawn by the OpenGL renderer instead
(path tracer disabled or failed, 2D views and subviews such as mirrors,
remote cameras and portal skies).
========================
*/
bool RB_PT_DrawView( void ) {
	const viewDef_t *viewDef = backEnd.viewDef;
	if ( !r_ptDraw.GetBool() || !viewDef || !viewDef->viewEntitys || viewDef->isSubview ) {
		return false;
	}
	if ( !PT_EnsureBackend() ) {
		return false;
	}

	if ( !pt.backend->Resize( glConfig.vidWidth, glConfig.vidHeight ) ) {
		return false;
	}

	// build and upload the scene
	pt.scene->BeginFrame();
	for ( int i = 0; i < viewDef->numDrawSurfs; i++ ) {
		pt.scene->AddDrawSurface( viewDef->drawSurfs[i] );
	}
	for ( const viewLight_t *vLight = viewDef->viewLights; vLight; vLight = vLight->next ) {
		pt.scene->AddViewLight( vLight );
	}
	pt.scene->EndFrame();

	const bool uploaded = pt.backend->UploadScene( *pt.scene );
	pt.scene->ReleasePendingTextures();
	if ( !uploaded ) {
		// a failed upload loses the pending textures, start over with a clean cache
		pt.scene->Clear();
		return false;
	}

	// trace resolution follows the viewport, not the window
	const int vpWidth = viewDef->viewport.x2 + 1 - viewDef->viewport.x1;
	const int vpHeight = viewDef->viewport.y2 + 1 - viewDef->viewport.y1;
	if ( vpWidth <= 0 || vpHeight <= 0 ) {
		return false;
	}
	const float scale = r_ptRenderScale.GetFloat();
	const int traceWidth = idMath::ClampInt( Min( vpWidth, PT_MIN_RENDER_SIZE ), Min( vpWidth, glConfig.vidWidth ), idMath::Ftoi( vpWidth * scale ) );
	const int traceHeight = idMath::ClampInt( Min( vpHeight, PT_MIN_RENDER_SIZE ), Min( vpHeight, glConfig.vidHeight ), idMath::Ftoi( vpHeight * scale ) );

	ptRenderParams_t params;
	PT_SetupParams( params, viewDef, traceWidth, traceHeight );

	const float lightScale = Max( 0.0f, r_lightScale.GetFloat() );
	bool reset = !r_ptAccumulation.GetBool() || !pt.havePrev || PT_ViewChanged( params, pt.prevParams )
		|| lightScale != pt.prevLightScale;
	switch ( r_ptResetOnSceneChange.GetInteger() ) {
		case 2:		reset |= pt.scene->ContentHash() != pt.prevContentHash;		// fall through
		case 1:		reset |= pt.scene->StructureHash() != pt.prevStructureHash;	break;
		default:	break;
	}
	if ( reset ) {
		pt.backend->ResetAccumulation();
	}
	pt.prevParams = params;
	pt.prevContentHash = pt.scene->ContentHash();
	pt.prevStructureHash = pt.scene->StructureHash();
	pt.prevLightScale = lightScale;
	pt.havePrev = true;

	PT_ApplyProgressiveQuality( params, pt.backend->GetAccumulatedFrames() );

	if ( !pt.backend->Render( params ) ) {
		return false;
	}

	pt.pixels.SetNum( traceWidth * traceHeight * 4, false );
	if ( !pt.backend->ReadPixels( pt.pixels.Ptr(), traceWidth, traceHeight ) ) {
		return false;
	}

	RB_BeginDrawingView();
	PT_BlitToViewport( pt.pixels.Ptr(), traceWidth, traceHeight );

	if ( r_ptDebug.GetBool() ) {
		ptBackendStats_t stats;
		pt.backend->GetStats( stats );
		common->Printf( "pt: %dx%d, %d tris, %d lights, %d frames, %.2f ms\n", traceWidth, traceHeight,
			stats.numTriangles, stats.numLights, pt.backend->GetAccumulatedFrames(), stats.traceMs );
	}
	return true;
}

#endif // HAVE_PATHTRACER
