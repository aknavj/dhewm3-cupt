#include "sys/platform.h"

#ifdef PT_PHYSICAL

#include "framework/Common.h"
#include "framework/CmdSystem.h"
#include "renderer/tr_local.h"
#include "renderer_pt/pt_physical.h"
#include "renderer_pt/pt_render.h"
#include "sys/sys_imgui.h"

extern idCVar r_ptDraw;
extern idCVar r_ptMode;
extern idCVar r_ptToneMapMode;
extern idCVar r_ptDenoise;

static idCVar r_ptPhysicalMetersPerUnit( "r_ptPhysicalMetersPerUnit", "0.0254", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"physical scale: meters per Doom unit (default assumes inches)", 0.001f, 1.0f );
static idCVar r_ptPhysicalLightLumens( "r_ptPhysicalLightLumens", "1500", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"lumens represented by a unit-white engine light stage", 1.0f, 1000000.0f );
static idCVar r_ptPhysicalSunLux( "r_ptPhysicalSunLux", "50000", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"illuminance in lux represented by a unit-white directional light", 1.0f, 150000.0f );
static idCVar r_ptPhysicalEmissionNits( "r_ptPhysicalEmissionNits", "100", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"cd/m^2 represented by unit-white material emission", 0.0f, 100000.0f );
static idCVar r_ptPhysicalIor( "r_ptPhysicalIor", "1.5", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"opaque dielectric index of refraction; legacy specular RGB is not F0", 1.01f, 2.5f );
static idCVar r_ptPhysicalRoughnessScale( "r_ptPhysicalRoughnessScale", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"scale for approximate roughness derived from legacy specular maps", 0.1f, 2.0f );
static idCVar r_ptPhysicalISO( "r_ptPhysicalISO", "800", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"photographic ISO sensitivity", 25.0f, 25600.0f );
static idCVar r_ptPhysicalShutter( "r_ptPhysicalShutter", "0.03333333", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"shutter time in seconds, exposure only (no motion blur)", 0.000125f, 30.0f );
static idCVar r_ptPhysicalFNumber( "r_ptPhysicalFNumber", "2.8", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"lens f-number for exposure and depth of field", 1.0f, 32.0f );
static idCVar r_ptPhysicalExposureCompensation( "r_ptPhysicalExposureCompensation", "0", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"photographic exposure compensation in stops", -10.0f, 10.0f );
static idCVar r_ptPhysicalFocalLength( "r_ptPhysicalFocalLength", "35", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"lens focal length in millimeters; determines horizontal field of view", 10.0f, 200.0f );
static idCVar r_ptPhysicalSensorWidth( "r_ptPhysicalSensorWidth", "36", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"sensor width in millimeters; sensor height follows the viewport aspect", 8.0f, 70.0f );
static idCVar r_ptPhysicalFocusDistance( "r_ptPhysicalFocusDistance", "3", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_FLOAT,
	"focus plane distance in meters along camera forward", 0.25f, 1000.0f );
static idCVar r_ptPhysicalDOF( "r_ptPhysicalDOF", "1", CVAR_RENDERER | CVAR_ARCHIVE | CVAR_BOOL,
	"thin-lens depth of field (off keeps photographic exposure with a pinhole)" );

static void PT_PhysicalPanel_f( const idCmdArgs &args ) {
	using namespace D3::ImGuiHooks;
	if ( !IsImguiEnabled() ) {
		common->Warning( "Physical PT controls require an initialized ImGui context" );
		return;
	}
	if ( GetOpenWindowsMask() & D3_ImGuiWin_PhysicalPT ) {
		CloseWindow( D3_ImGuiWin_PhysicalPT );
	} else {
		OpenWindow( D3_ImGuiWin_PhysicalPT );
	}
}

void R_PT_InitPhysical() {
	cmdSystem->AddCommand( "r_ptPhysicalUI", PT_PhysicalPanel_f, CMD_FL_RENDERER, "toggle experimental physical path tracer controls" );
	common->Printf( "PT_PHYSICAL: experimental photometric lighting and thin-lens camera enabled; r_ptPhysicalUI opens controls\n" );
}

void R_PT_PhysicalCamera( viewDef_t *view ) {
	if ( view->isSubview || !r_ptDraw.GetBool() || r_ptMode.GetInteger() != PT_MODE_PATHTRACE ) return;
	const float width = view->viewport.x2 + 1 - view->viewport.x1;
	const float height = view->viewport.y2 + 1 - view->viewport.y1;
	if ( width <= 0.0f || height <= 0.0f ) return;
	const float sensorWidth = r_ptPhysicalSensorWidth.GetFloat();
	const float focalLength = r_ptPhysicalFocalLength.GetFloat();
	view->renderView.fov_x = PT_PhysicalFov( sensorWidth, focalLength );
	view->renderView.fov_y = PT_PhysicalFov( sensorWidth * height / width, focalLength );
}

void R_PT_PhysicalParams( ptRenderParams_t &p ) {
	if ( p.renderMode != PT_MODE_PATHTRACE ) return;
	p.physicalMetersPerUnit = r_ptPhysicalMetersPerUnit.GetFloat();
	p.physicalLightLumens = r_ptPhysicalLightLumens.GetFloat();
	p.physicalSunLux = r_ptPhysicalSunLux.GetFloat();
	p.physicalIor = r_ptPhysicalIor.GetFloat();
	p.physicalRoughnessScale = r_ptPhysicalRoughnessScale.GetFloat();
	p.physicalApertureRadius = r_ptPhysicalDOF.GetBool()
		? PT_PhysicalApertureRadius( r_ptPhysicalFocalLength.GetFloat(), r_ptPhysicalFNumber.GetFloat(), p.physicalMetersPerUnit ) : 0.0f;
	p.physicalFocusDistance = r_ptPhysicalFocusDistance.GetFloat() / p.physicalMetersPerUnit;
	p.exposure = PT_PhysicalExposure( r_ptPhysicalFNumber.GetFloat(), r_ptPhysicalShutter.GetFloat(),
		r_ptPhysicalISO.GetFloat(), r_ptPhysicalExposureCompensation.GetFloat() );
	p.emissionBoost = r_ptPhysicalEmissionNits.GetFloat();
	p.specularBoost = 1.0f;
	p.softShadowScale = 0.0f;
	p.fireflyClamp = 0.0f;
	p.indirectProb = 0.0f;
	p.volDensity = 0.0f;
}

static void PT_PhysicalSlider( const char *label, idCVar &var, float minimum, float maximum, const char *format, bool logarithmic = false ) {
	float value = var.GetFloat();
	if ( ImGui::SliderFloat( label, &value, minimum, maximum, format,
		ImGuiSliderFlags_AlwaysClamp | ( logarithmic ? ImGuiSliderFlags_Logarithmic : 0 ) ) ) {
		var.SetFloat( value );
	}
}

void R_PT_DrawPhysicalPanel() {
	using namespace D3::ImGuiHooks;
	bool open = true;
	ImGui::SetNextWindowSize( ImVec2( 510, 650 ), ImGuiCond_FirstUseEver );
	if ( ImGui::Begin( "Physical Path Tracer (experimental)", &open ) ) {
		ImGui::PushItemWidth( ImGui::GetContentRegionAvail().x * 0.5f );
		ImGui::TextWrapped( "PT_PHYSICAL is a build-time model, not a Doom 3 lighting preset. Controls are saved as CVars." );
		if ( !r_ptDraw.GetBool() || r_ptMode.GetInteger() != PT_MODE_PATHTRACE ) {
			ImGui::TextWrapped( "Select path tracing to preview these settings." );
			if ( ImGui::Button( "Use physical path tracing" ) ) {
				r_ptDraw.SetBool( true );
				r_ptMode.SetInteger( PT_MODE_PATHTRACE );
			}
		}
		const bool capturing = RB_PT_GetCaptureTargetFrames() > 0;
		if ( capturing ) ImGui::TextWrapped( "Capture in progress: controls locked until capture completes or r_ptAbort." );
		ImGui::BeginDisabled( capturing );
		if ( ImGui::CollapsingHeader( "Photographic camera", ImGuiTreeNodeFlags_DefaultOpen ) ) {
			PT_PhysicalSlider( "ISO", r_ptPhysicalISO, 25, 25600, "%.0f", true );
			PT_PhysicalSlider( "Shutter (seconds)", r_ptPhysicalShutter, 0.000125f, 30, "%.5f s", true );
			PT_PhysicalSlider( "Aperture", r_ptPhysicalFNumber, 1, 32, "f/%.1f", true );
			PT_PhysicalSlider( "Compensation", r_ptPhysicalExposureCompensation, -10, 10, "%+.2f EV" );
			PT_PhysicalSlider( "Focal length", r_ptPhysicalFocalLength, 10, 200, "%.1f mm" );
			PT_PhysicalSlider( "Sensor width", r_ptPhysicalSensorWidth, 8, 70, "%.1f mm" );
			bool dof = r_ptPhysicalDOF.GetBool();
			if ( ImGui::Checkbox( "Depth of field", &dof ) ) r_ptPhysicalDOF.SetBool( dof );
			PT_PhysicalSlider( "Focus distance", r_ptPhysicalFocusDistance, 0.25f, 1000, "%.2f m", true );
			ImGui::Text( "EV100: %.2f   horizontal FOV: %.1f degrees",
				PT_PhysicalEV100( r_ptPhysicalFNumber.GetFloat(), r_ptPhysicalShutter.GetFloat(), r_ptPhysicalISO.GetFloat() ),
				PT_PhysicalFov( r_ptPhysicalSensorWidth.GetFloat(), r_ptPhysicalFocalLength.GetFloat() ) );
			ImGui::TextWrapped( "Shutter controls exposure, not motion blur. Lens blur converges progressively; real-time temporal denoising is disabled with DOF." );
		}
		if ( ImGui::CollapsingHeader( "Asset calibration", ImGuiTreeNodeFlags_DefaultOpen ) ) {
			PT_PhysicalSlider( "Light flux", r_ptPhysicalLightLumens, 1, 1000000, "%.0f lm", true );
			PT_PhysicalSlider( "Directional illuminance", r_ptPhysicalSunLux, 1, 150000, "%.0f lux", true );
			PT_PhysicalSlider( "Emissive luminance", r_ptPhysicalEmissionNits, 0, 100000, "%.1f cd/m^2", true );
			PT_PhysicalSlider( "World scale", r_ptPhysicalMetersPerUnit, 0.001f, 1, "%.4f m/unit", true );
			PT_PhysicalSlider( "Dielectric IOR", r_ptPhysicalIor, 1.01f, 2.5f, "%.2f" );
			PT_PhysicalSlider( "Roughness scale", r_ptPhysicalRoughnessScale, 0.1f, 2, "%.2f" );
			ImGui::TextWrapped( "Doom 3 assets are not measured PBR data. Specular maps approximate roughness, not metalness. Legacy light volumes, boosts, firefly clamps and volumetrics are not used." );
		}
		int toneMap = r_ptToneMapMode.GetInteger();
		if ( ImGui::Combo( "Tone mapping", &toneMap, "Reinhard\0ACES\0Uncharted 2\0" ) ) r_ptToneMapMode.SetInteger( toneMap );
		bool denoise = r_ptDenoise.GetBool();
		if ( ImGui::Checkbox( "Denoise", &denoise ) ) r_ptDenoise.SetBool( denoise );
		ImGui::EndDisabled();
		ImGui::PopItemWidth();
	}
	ImGui::End();
	if ( !open ) CloseWindow( D3_ImGuiWin_PhysicalPT );
}

#endif
