#ifndef __PT_SCENE_H__
#define __PT_SCENE_H__

#include "idlib/containers/List.h"
#include "idlib/containers/HashIndex.h"
#include "renderer_pt/pt_types.h"

class idMaterial;
class idImage;
struct drawSurf_s;
struct viewLight_s;

// FNV-1a over 32 bit words
static ID_INLINE unsigned int PT_HashWords( unsigned int h, const void *data, size_t bytes ) {
	const unsigned int *w = (const unsigned int *)data;
	const size_t n = bytes / sizeof( unsigned int );
	for ( size_t i = 0; i < n; i++ ) {
		h = ( h ^ w[i] ) * 16777619u;
	}
	return h;
}

// decodes a gamma space color factor to linear, the inverse of the default r_ptGamma
static ID_INLINE float PT_SrgbToLinear( float c ) {
	return c > 0.0f ? powf( c, 2.2f ) : 0.0f;
}

typedef struct {
	int				index;			// slot in idPathTracerScene::Textures()
	int				width;
	int				height;
	unsigned char *	rgba;			// width * height * 4 bytes, owned by the scene
} ptTextureUpload_t;

/*
===============================================================================

	idPathTracerScene

	CPU-side, compute-API agnostic snapshot of one view. Converts engine draw
	surfaces, materials, images and lights into the POD types of pt_types.h.

	Geometry and lights are rebuilt every view. Materials and textures are cached
	across views (keyed by engine pointer) until the texture budget overflows or
	Clear() is called. Materials with entity or time driven registers get one
	slot per distinct register set used in a frame, so entities sharing such a
	material keep their own colors. Newly referenced textures are read back from OpenGL and
	queued as pending uploads for the backend.

===============================================================================
*/
class idPathTracerScene {
public:
							idPathTracerScene();
							~idPathTracerScene();

	// drops all cached materials/textures, e.g. after a backend change
	void					Clear();

	void					BeginFrame();
	void					AddDrawSurface( const struct drawSurf_s *surf );
	void					AddViewLight( const struct viewLight_s *vLight );
	void					EndFrame();

	const idList<ptVertex_t> &		Vertices() const { return vertices; }
	const idList<ptTriangle_t> &	Triangles() const { return triangles; }
	const idList<ptMaterial_t> &	Materials() const { return materials; }
	const idList<ptLight_t> &		Lights() const { return lights; }
	const idList<ptTextureInfo_t> &	Textures() const { return textures; }

	// textures read back since the last ReleasePendingTextures()
	const idList<ptTextureUpload_t> & PendingTextures() const { return pendingTextures; }
	void					ReleasePendingTextures();

	// bumped whenever cached textures are flushed; backends must drop their copies
	int						TextureGeneration() const { return textureGeneration; }

	// hash of vertex positions, light parameters and material values; changes
	// with any animation (skinned meshes, time driven shader registers, ...)
	unsigned int			ContentHash() const { return contentHash; }

	// hash of the scene structure only: triangle, light, material and texture
	// counts and the texture generation; changes when things appear or vanish
	unsigned int			StructureHash() const { return structureHash; }

	// scene bounds (min xyz, max xyz) of all triangles
	const float *			Bounds() const { return bounds; }

private:
	int						GetOrAddMaterial( const idMaterial *material, const float *shaderRegisters );
	void					BuildMaterial( ptMaterial_t &mat, idVec3 &emission, const idMaterial *material, const float *shaderRegisters );
	int						AddTexture( const idImage *image );
	void					FlushMaterialsAndTextures();
	void					AddDefaultMaterial();

	// triFlags: PT_TRI_WEAPON / PT_TRI_NO_OCCLUDE from the entity, material derived flags are added
	void					AddTriangles( const idDrawVert *verts, int numVerts, const int *indexes, int numIndexes,
										  int materialIndex, const float *modelMatrix, int triFlags );
	bool					AddEmissiveAreaLight( const ptVertex_t &v0, const ptVertex_t &v1, const ptVertex_t &v2, int materialIndex );
	// a non-NULL lightProject selects the GL style box falloff (light image * falloff image)
	void					AddLight( const idVec3 &position, const idVec3 &color, float intensity, int type,
									  float lightRadius, const float *lightProject, int projTexIndex, int falloffTexIndex,
									  const float *lightRadiusXYZ );
	void					AddSpotLight( const idVec3 &position, const idVec3 &direction, const idVec3 &color,
										  float intensity, float coneAngle, float coneFalloff, float lightRadius,
										  const float *lightProject, int projTexIndex, int falloffTexIndex,
										  const float *rightAxis, const float *upAxis );

	// per view
	idList<ptVertex_t>		vertices;
	idList<ptTriangle_t>	triangles;
	idList<ptLight_t>		lights;
	idList<ptLight_t>		areaLights;			// emissive triangles, appended after real lights in EndFrame
	idList<int>				areaLightTris;		// triangle of each areaLights entry
	float					bounds[6];
	unsigned int			contentHash;
	unsigned int			structureHash;
	int						frameCount;

	// cached across views
	idList<ptMaterial_t>	materials;
	idList<idVec3>			materialEmission;
	idList<const idMaterial *> materialPtrs;
	idList<int>				materialFrame;		// frame the material was last evaluated
	idList<unsigned int>	materialRegHash;	// shader registers the slot was built with (animated materials)
	idHashIndex				materialHash;

	idList<ptTextureInfo_t>	textures;
	idList<const idImage *>	texturePtrs;
	idHashIndex				textureHash;
	idList<ptTextureUpload_t> pendingTextures;
	int						textureGeneration;
	bool					needTextureFlush;
};

#endif /* !__PT_SCENE_H__ */
