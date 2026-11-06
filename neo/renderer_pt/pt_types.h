#ifndef __PT_TYPES_H__
#define __PT_TYPES_H__

/*
===============================================================================

	Path tracer shared data types.

	Everything in this header is consumed by the API-agnostic host code AND by
	every compute backend's device code (CUDA today, OpenCL / Vulkan compute
	tomorrow). To keep the memory layout identical across all of them:

	  - plain C only (no classes, references, templates, bool)
	  - no pointers (device addresses are backend specific)
	  - only 4-byte scalars, so there is no implicit padding

===============================================================================
*/

#define PT_MAX_TEXTURES			2048
#define PT_MAX_MATERIALS		1024
#define PT_MAX_LIGHTS			512
#define PT_MAX_TEXTURE_SIZE		4096
#define PT_MIN_RENDER_SIZE		64

#define PT_DEFAULT_MATERIAL		0		// slot 0 is always the fallback material

enum ptLightType_t {
	PT_LIGHT_POINT			= 0,	// point light, optionally a spot light when coneAngle > 0
	PT_LIGHT_DIRECTIONAL	= 1,
	PT_LIGHT_AREA			= 2		// generated from emissive triangles
};

// ptTriangle_t::flags, derived from the material and the entity on the host
enum ptTriangleFlags_t {
	PT_TRI_WEAPON			= 1,	// view weapon (weaponDepthHack): seen by camera rays only, never occludes
	PT_TRI_NO_OCCLUDE		= 2,	// never blocks shadow rays (noShadows, additive, unlit overlays)
	PT_TRI_OPAQUE			= 4,	// always blocks shadow rays, no material lookup needed
	PT_TRI_AREA_LIGHT		= 8		// also a PT_LIGHT_AREA, bounce hits skip its emission where it was sampled
};

enum ptBlendMode_t {
	PT_BLEND_OPAQUE			= 0,
	PT_BLEND_ALPHA			= 1,
	PT_BLEND_ADD			= 2,
	PT_BLEND_MODULATE		= 3
};

enum ptRenderMode_t {
	PT_MODE_PATHTRACE		= 0,
	PT_MODE_BVH_DEBUG		= 1,
	PT_MODE_TEX_COMBINED	= 2,
	PT_MODE_TEX_BASECOLOR	= 3,
	PT_MODE_TEX_ALBEDO		= 4,
	PT_MODE_TEX_NORMAL		= 5,
	PT_MODE_TEX_SPECULAR	= 6,
	PT_MODE_WHITTED			= 7,
	PT_MODE_COUNT
};

enum ptToneMap_t {
	PT_TONEMAP_REINHARD		= 0,
	PT_TONEMAP_ACES			= 1,
	PT_TONEMAP_UNCHARTED2	= 2
};

typedef struct {
	float position[3];
	float normal[3];
	float texcoord[2];
	float tangent[3];
	float binormal[3];
	float color[4];				// vertex color RGBA
} ptVertex_t;

typedef struct {
	int vertexIndices[3];
	float bounds[6];			// min xyz, max xyz
	int materialIndex;
	int flags;					// ptTriangleFlags_t
} ptTriangle_t;

typedef struct {
	int width;
	int height;					// 0x0 means "not resident", samplers return white
} ptTextureInfo_t;

/*
Colors (albedo, emission, light colors) are linear. Textures are stored as read
back from GL; albedo, emission and light textures are sRGB encoded and decoded
by the samplers, normal and specular maps are used as is.
*/
typedef struct {
	float albedo[4];			// base color + alpha
	float emission[3];			// emissive color (glowing materials), modulated by emissionTexture
	float metallic;
	float roughness;
	float ior;					// index of refraction
	float transmission;			// glass etc.
	float alphaTest;			// 0 = no test, >0 = threshold
	int blendMode;				// ptBlendMode_t
	int useVertexColor;
	int isAmbientOnly;			// no interaction stages, rendered as unlit overlay
	int noShadows;				// does not cast shadows
	int albedoTexture;			// texture index or -1
	int normalTexture;
	int specularTexture;
	int emissionTexture;		// glow stage texture or -1
	float bumpScale;
	float texTransform[6];		// scale.x, scale.y, rotate, translate.x, translate.y, scroll_speed
	float polygonOffset;		// depth bias for decals
} ptMaterial_t;

typedef struct {
	int type;					// ptLightType_t
	float position[3];
	float direction[3];			// normalized forward (spot / directional)
	float color[3];
	float intensity;
	float radius;				// falloff boundary
	float area[3];				// emissive triangle area
	int textureIndex;			// light texture or -1
	float volumetric;			// volumetric scattering intensity (0 = off)
	float coneAngle;			// spot half angle in radians (0 = omni)
	float coneFalloff;			// spot edge exponent

	float lightProject[4][4];	// Doom 3 light projection planes (world space)
	int projectedTextureIndex;	// light image, sampled at the projected (s, t)
	int falloffTextureIndex;	// falloff image, sampled at (falloff, 0.5)
	int boxFalloff;				// attenuate by lightProject / light image / falloff image
								// like the GL renderer instead of the spherical window

	float lightRadius3[3];		// xyz extents for area sampling

	float right[3];				// projected light axes (normalized)
	float up[3];
} ptLight_t;

typedef struct {
	float bounds[6];			// min xyz, max xyz
	int leftChild;				// -1 for leaves
	int rightChild;
	int firstPrimitive;			// leaves only
	int primitiveCount;
} ptBVHNode_t;

/*
================
ptRenderParams_t

Everything a backend needs to trace one frame. Passed by value to kernels, so
it must stay small (well under the 4 KB kernel parameter limit) and POD.
================
*/
typedef struct {
	// camera
	float camPos[3];
	float camForward[3];
	float camRight[3];
	float camUp[3];
	float fovX;
	float fovY;

	// trace resolution, must be <= the size passed to idPathTracerBackend::Resize
	int width;
	int height;

	int renderMode;				// ptRenderMode_t
	int frameIndex;				// filled in by the backend (accumulated frame count)

	// sampling
	int samplesPerPixel;
	int maxDepth;
	int maxLightSamples;
	float indirectProb;

	// path termination
	int rrEnabled;
	int rrMinBounces;
	float rrSurvivalMin;
	float earlyTermThreshold;	// Russian roulette below this throughput (unbiased)
	float throughputClamp;		// Russian roulette below this throughput after shading (unbiased)
	float fireflyClamp;

	// shading
	float rayOffset;
	float emissionBoost;
	float specularBoost;
	float softShadowScale;

	// sky
	float skyIntensity;
	float skyZenith[3];
	float skyHorizon[3];
	float skyGround[3];

	// volumetrics
	float volDensity;
	int volSteps;
	float volAnisotropy;
	float volFalloff;
	float volMaxDist;

	// display
	float exposure;
	float gamma;
	int toneMapMode;			// ptToneMap_t
} ptRenderParams_t;

#endif /* !__PT_TYPES_H__ */
