#include "sys/platform.h"
#include "renderer/tr_local.h"
#include "renderer_pt/pt_scene.h"

#include <new>

/*
===============================================================================

	idPathTracerScene

	Converts the engine's view (draw surfaces + view lights) into flat,
	pointer-free arrays that any compute backend can upload as-is.

===============================================================================
*/

extern idCVar r_ptDebug;
extern idCVar r_ptVolumetric;

static const float PT_AREA_LIGHT_MIN_EMISSION	= 0.5f;

/*
========================
PT_Reserve

idList grows linearly by its granularity, which is quadratic for the
hundreds of thousands of vertices appended per frame. Grow geometrically.
========================
*/
template< class T >
static void PT_Reserve( idList<T> &list, int needed ) {
	if ( needed > list.NumAllocated() ) {
		list.Resize( Max( needed, list.NumAllocated() * 2 ) );
	}
}

static ID_INLINE void PT_TransformPoint( const float *m, const float *in, float *out ) {
	out[0] = m[0] * in[0] + m[4] * in[1] + m[8]  * in[2] + m[12];
	out[1] = m[1] * in[0] + m[5] * in[1] + m[9]  * in[2] + m[13];
	out[2] = m[2] * in[0] + m[6] * in[1] + m[10] * in[2] + m[14];
}

static ID_INLINE void PT_TransformVector( const float *m, const float *in, float *out ) {
	out[0] = m[0] * in[0] + m[4] * in[1] + m[8]  * in[2];
	out[1] = m[1] * in[0] + m[5] * in[1] + m[9]  * in[2];
	out[2] = m[2] * in[0] + m[6] * in[1] + m[10] * in[2];
}

static ID_INLINE void PT_NormalizeOr( float *v, float dx, float dy, float dz ) {
	const float len = sqrtf( v[0] * v[0] + v[1] * v[1] + v[2] * v[2] );
	if ( len > 1e-6f ) {
		const float inv = 1.0f / len;
		v[0] *= inv; v[1] *= inv; v[2] *= inv;
	} else {
		v[0] = dx; v[1] = dy; v[2] = dz;
	}
}

/*
========================
idPathTracerScene::idPathTracerScene
========================
*/
idPathTracerScene::idPathTracerScene() {
	contentHash = 0;
	structureHash = 0;
	frameCount = 0;
	textureGeneration = 0;
	needTextureFlush = false;
	bounds[0] = bounds[1] = bounds[2] = bounds[3] = bounds[4] = bounds[5] = 0.0f;
	AddDefaultMaterial();
}

/*
========================
idPathTracerScene::~idPathTracerScene
========================
*/
idPathTracerScene::~idPathTracerScene() {
	ReleasePendingTextures();
}

/*
========================
idPathTracerScene::Clear
========================
*/
void idPathTracerScene::Clear() {
	vertices.Clear();
	triangles.Clear();
	lights.Clear();
	areaLights.Clear();
	areaLightTris.Clear();
	FlushMaterialsAndTextures();
}

/*
========================
idPathTracerScene::FlushMaterialsAndTextures
========================
*/
void idPathTracerScene::FlushMaterialsAndTextures() {
	ReleasePendingTextures();

	textures.Clear();
	texturePtrs.Clear();
	textureNameHashes.Clear();
	textureGLNames.Clear();
	textureHash.Free();

	materials.Clear();
	materialEmission.Clear();
	materialCache.Clear();
	AddDefaultMaterial();

	textureGeneration++;
	needTextureFlush = false;
}

/*
========================
idPathTracerScene::ReleasePendingTextures
========================
*/
void idPathTracerScene::ReleasePendingTextures() {
	for ( int i = 0; i < pendingTextures.Num(); i++ ) {
		delete[] pendingTextures[i].rgba;
	}
	pendingTextures.Clear();
}

/*
========================
idPathTracerScene::BeginFrame
========================
*/
void idPathTracerScene::BeginFrame() {
	frameCount++;
	materialCache.BeginFrame();

	// keep the allocations, the scene is rebuilt every view
	vertices.SetNum( 0, false );
	triangles.SetNum( 0, false );
	lights.SetNum( 0, false );
	areaLights.SetNum( 0, false );
	areaLightTris.SetNum( 0, false );

	if ( needTextureFlush ) {
		common->Printf( "idPathTracerScene: flushing %d textures and %d materials (texture limit reached)\n",
			textures.Num(), materials.Num() );
		FlushMaterialsAndTextures();
	}

	contentHash = 2166136261u;
}

/*
========================
idPathTracerScene::AddDrawSurface
========================
*/
void idPathTracerScene::AddDrawSurface( const drawSurf_t *surf ) {
	if ( !surf || !surf->geo ) {
		return;
	}
	const srfTriangles_t *tri = surf->geo;
	if ( !tri->verts || !tri->indexes || tri->numVerts <= 0 || tri->numIndexes < 3 ) {
		return;
	}

	const int materialIndex = GetOrAddMaterial( surf->material, surf->shaderRegisters );
	const float *modelMatrix = surf->space ? surf->space->modelMatrix : NULL;

	int triFlags = 0;
	if ( surf->space ) {
		if ( surf->space->weaponDepthHack ) {
			triFlags |= PT_TRI_WEAPON;
		}
		if ( surf->space->entityDef && surf->space->entityDef->parms.noShadow ) {
			triFlags |= PT_TRI_NO_OCCLUDE;
		}
	}

	AddTriangles( tri->verts, tri->numVerts, tri->indexes, tri->numIndexes, materialIndex, modelMatrix, triFlags );
}

/*
========================
idPathTracerScene::AddTriangles
========================
*/
void idPathTracerScene::AddTriangles( const idDrawVert *verts, int numVerts, const int *indexes, int numIndexes,
									  int materialIndex, const float *modelMatrix, int triFlags ) {
	const int vertexBase = vertices.Num();

	PT_Reserve( vertices, vertexBase + numVerts );
	for ( int i = 0; i < numVerts; i++ ) {
		const idDrawVert &dv = verts[i];
		ptVertex_t v;

		if ( modelMatrix ) {
			PT_TransformPoint( modelMatrix, dv.xyz.ToFloatPtr(), v.position );
			PT_TransformVector( modelMatrix, dv.normal.ToFloatPtr(), v.normal );
			PT_TransformVector( modelMatrix, dv.tangents[0].ToFloatPtr(), v.tangent );
			PT_TransformVector( modelMatrix, dv.tangents[1].ToFloatPtr(), v.binormal );
		} else {
			memcpy( v.position, dv.xyz.ToFloatPtr(), sizeof( v.position ) );
			memcpy( v.normal, dv.normal.ToFloatPtr(), sizeof( v.normal ) );
			memcpy( v.tangent, dv.tangents[0].ToFloatPtr(), sizeof( v.tangent ) );
			memcpy( v.binormal, dv.tangents[1].ToFloatPtr(), sizeof( v.binormal ) );
		}
		PT_NormalizeOr( v.normal, 0.0f, 0.0f, 1.0f );
		PT_NormalizeOr( v.tangent, 1.0f, 0.0f, 0.0f );
		PT_NormalizeOr( v.binormal, 0.0f, 1.0f, 0.0f );

		v.texcoord[0] = dv.st[0];
		v.texcoord[1] = dv.st[1];

		const float inv255 = 1.0f / 255.0f;
		v.color[0] = dv.color[0] * inv255;
		v.color[1] = dv.color[1] * inv255;
		v.color[2] = dv.color[2] * inv255;
		v.color[3] = dv.color[3] * inv255;

		vertices.Append( v );
		contentHash = PT_HashWords( contentHash, v.position, sizeof( v.position ) );
	}
	contentHash = PT_HashWords( contentHash, &materialIndex, sizeof( materialIndex ) );

	const ptMaterial_t &mat = materials[materialIndex];
	if ( mat.noShadows || mat.isAmbientOnly || mat.blendMode == PT_BLEND_ADD ) {
		triFlags |= PT_TRI_NO_OCCLUDE;
	}
	if ( !( triFlags & PT_TRI_NO_OCCLUDE ) && mat.alphaTest <= 0.0f && mat.transmission <= 0.0f && mat.blendMode != PT_BLEND_ALPHA ) {
		triFlags |= PT_TRI_OPAQUE;
	}

	const bool emissive = materialIndex != PT_DEFAULT_MATERIAL
		&& !( triFlags & PT_TRI_WEAPON )
		&& materials[materialIndex].blendMode == PT_BLEND_OPAQUE
		&& !materials[materialIndex].isAmbientOnly
		&& ( materialEmission[materialIndex].x + materialEmission[materialIndex].y + materialEmission[materialIndex].z ) > PT_AREA_LIGHT_MIN_EMISSION;

	PT_Reserve( triangles, triangles.Num() + numIndexes / 3 );
	for ( int i = 0; i + 2 < numIndexes; i += 3 ) {
		const int i0 = indexes[i + 0];
		const int i1 = indexes[i + 1];
		const int i2 = indexes[i + 2];
		if ( i0 < 0 || i0 >= numVerts || i1 < 0 || i1 >= numVerts || i2 < 0 || i2 >= numVerts ) {
			if ( r_ptDebug.GetBool() ) {
				common->Warning( "idPathTracerScene: skipping triangle with out of range index\n" );
			}
			continue;
		}

		ptTriangle_t tri;
		tri.vertexIndices[0] = vertexBase + i0;
		tri.vertexIndices[1] = vertexBase + i1;
		tri.vertexIndices[2] = vertexBase + i2;
		tri.materialIndex = materialIndex;
		tri.flags = triFlags;

		const ptVertex_t &v0 = vertices[tri.vertexIndices[0]];
		const ptVertex_t &v1 = vertices[tri.vertexIndices[1]];
		const ptVertex_t &v2 = vertices[tri.vertexIndices[2]];
		for ( int a = 0; a < 3; a++ ) {
			tri.bounds[a]     = Min( v0.position[a], Min( v1.position[a], v2.position[a] ) );
			tri.bounds[a + 3] = Max( v0.position[a], Max( v1.position[a], v2.position[a] ) );
		}
		const int triIndex = triangles.Append( tri );

		if ( emissive && AddEmissiveAreaLight( v0, v1, v2, materialIndex ) ) {
			triangles[triIndex].flags |= PT_TRI_AREA_LIGHT;
			areaLightTris.Append( triIndex );
		}
	}
}

/*
========================
idPathTracerScene::AddEmissiveAreaLight
========================
*/
bool idPathTracerScene::AddEmissiveAreaLight( const ptVertex_t &v0, const ptVertex_t &v1, const ptVertex_t &v2, int materialIndex ) {
	// engine lights take priority, area lights are appended into what is left in EndFrame
	if ( areaLights.Num() >= PT_MAX_LIGHTS ) {
		return false;
	}

	const idVec3 p0( v0.position[0], v0.position[1], v0.position[2] );
	const idVec3 p1( v1.position[0], v1.position[1], v1.position[2] );
	const idVec3 p2( v2.position[0], v2.position[1], v2.position[2] );
	const idVec3 center = ( p0 + p1 + p2 ) * ( 1.0f / 3.0f );
	const idVec3 normal = ( p1 - p0 ).Cross( p2 - p0 );
	const float area = normal.Length() * 0.5f;
	if ( area < 1e-4f ) {
		return false;
	}

	const idVec3 &emission = materialEmission[materialIndex];
	const float emissive = emission.x + emission.y + emission.z;

	ptLight_t light;
	memset( &light, 0, sizeof( light ) );
	light.type = PT_LIGHT_AREA;
	light.position[0] = center.x;
	light.position[1] = center.y;
	light.position[2] = center.z;
	light.direction[0] = normal.x / ( area * 2.0f );
	light.direction[1] = normal.y / ( area * 2.0f );
	light.direction[2] = normal.z / ( area * 2.0f );

	// clamped to avoid fireflies from tiny, very bright emitters
	light.color[0] = idMath::ClampFloat( 0.0f, 1.0f, emission.x );
	light.color[1] = idMath::ClampFloat( 0.0f, 1.0f, emission.y );
	light.color[2] = idMath::ClampFloat( 0.0f, 1.0f, emission.z );
	light.intensity = idMath::ClampFloat( 0.0f, 10.0f, emissive * idMath::Sqrt( area ) * 0.25f );
	light.radius = idMath::Sqrt( area );
	light.area[0] = area;
	light.textureIndex = materials[materialIndex].emissionTexture >= 0 ? materials[materialIndex].emissionTexture
																	   : materials[materialIndex].albedoTexture;
	light.projectedTextureIndex = -1;
	light.falloffTextureIndex = -1;
	light.lightRadius3[0] = light.lightRadius3[1] = light.lightRadius3[2] = light.radius;

	areaLights.Append( light );
	return true;
}

/*
========================
idPathTracerScene::AddViewLight
========================
*/
void idPathTracerScene::AddViewLight( const viewLight_t *vLight ) {
	if ( !vLight || !vLight->lightDef ) {
		return;
	}

	const idRenderLightLocal *lightDef = vLight->lightDef;
	const renderLight_t &parms = lightDef->parms;
	const idMaterial *lightShader = vLight->lightShader ? vLight->lightShader : lightDef->lightShader;
	if ( !lightShader ) {
		return;
	}

	// fog, ambient and blend lights do not cast direct light
	if ( lightShader->IsFogLight() || lightShader->IsAmbientLight() || lightShader->IsBlendLight() ) {
		return;
	}

	// like the GL interaction pass: the first enabled light stage gives the color
	// (shaderParms and any time driven flicker) and the light image
	const float *lightRegs = vLight->shaderRegisters;
	const shaderStage_t *lightStage = NULL;
	for ( int i = 0; i < lightShader->GetNumStages(); i++ ) {
		const shaderStage_t *stage = lightShader->GetStage( i );
		if ( stage && ( !lightRegs || lightRegs[stage->conditionRegister] != 0.0f ) ) {
			lightStage = stage;
			break;
		}
	}
	if ( lightShader->GetNumStages() > 0 && !lightStage ) {
		return;
	}

	idVec3 lightColor;
	if ( lightStage && lightRegs ) {
		for ( int c = 0; c < 3; c++ ) {
			lightColor[c] = Max( 0.0f, lightRegs[lightStage->color.registers[c]] );
		}
		if ( lightColor.x + lightColor.y + lightColor.z < 0.003f ) {
			return;
		}
	} else {
		lightColor.Set( parms.shaderParms[SHADERPARM_RED], parms.shaderParms[SHADERPARM_GREEN], parms.shaderParms[SHADERPARM_BLUE] );
		if ( lightColor.LengthSqr() < 0.01f ) {
			lightColor.Set( 1.0f, 1.0f, 1.0f );
		}
	}
	for ( int c = 0; c < 3; c++ ) {
		lightColor[c] = PT_SrgbToLinear( lightColor[c] );
	}

	const int lightImageIndex = ( lightStage && lightStage->texture.image ) ? AddTexture( lightStage->texture.image ) : -1;
	const int falloffImageIndex = lightShader->LightFalloffImage() ? AddTexture( lightShader->LightFalloffImage() ) : -1;

	float baseIntensity = parms.shaderParms[SHADERPARM_ALPHA];
	if ( baseIntensity < 0.01f ) {
		baseIntensity = 1.0f;
	}

	float projPlanes[4][4];
	for ( int p = 0; p < 4; p++ ) {
		projPlanes[p][0] = vLight->lightProject[p].Normal().x;
		projPlanes[p][1] = vLight->lightProject[p].Normal().y;
		projPlanes[p][2] = vLight->lightProject[p].Normal().z;
		projPlanes[p][3] = vLight->lightProject[p][3];
	}

	if ( parms.pointLight ) {
		const float radius = ( parms.lightRadius.x + parms.lightRadius.y + parms.lightRadius.z ) / 3.0f;
		// compensate for the BRDF's 1/pi normalization
		const float intensity = idMath::ClampFloat( 0.5f, 100.0f, baseIntensity * idMath::PI );

		// origin + axis * lightCenter
		const idVec3 &origin = lightDef->globalLightOrigin;

		AddLight( origin, lightColor, intensity, PT_LIGHT_POINT, radius, &projPlanes[0][0], lightImageIndex, falloffImageIndex,
			parms.lightRadius.ToFloatPtr() );
	} else if ( parms.parallel ) {
		idVec3 dir = parms.lightCenter;
		if ( dir.LengthSqr() < 0.01f ) {
			dir = parms.target - parms.origin;
		}
		dir.Normalize();

		AddLight( dir, lightColor, baseIntensity * 10.0f, PT_LIGHT_DIRECTIONAL, 0.0f, NULL, -1, -1, NULL );
	} else {
		const idVec3 worldTarget = parms.axis * parms.target;
		const idVec3 worldRight = parms.axis * parms.right;
		const idVec3 worldUp = parms.axis * parms.up;

		idVec3 lightDir = worldTarget;
		const float targetLen = lightDir.Length();
		if ( targetLen > 0.01f ) {
			lightDir /= targetLen;
		} else {
			lightDir.Set( 1.0f, 0.0f, 0.0f );
		}

		const float intensity = idMath::ClampFloat( 0.5f, 100.0f, baseIntensity * idMath::PI );
		const float halfWidth = Max( worldRight.Length(), worldUp.Length() );
		const float coneAngle = atanf( halfWidth / Max( targetLen, 1.0f ) );
		const float coneFalloff = 2.0f;

		float lightRange = ( parms.end - parms.start ).Length();
		if ( lightRange < 10.0f ) {
			lightRange = 500.0f;	// malformed frustum
		}

		idVec3 rightNorm = worldRight;
		rightNorm.Normalize();
		idVec3 upNorm = worldUp;
		upNorm.Normalize();

		AddSpotLight( parms.origin, lightDir, lightColor, intensity, coneAngle, coneFalloff, lightRange,
			&projPlanes[0][0], lightImageIndex, falloffImageIndex, rightNorm.ToFloatPtr(), upNorm.ToFloatPtr() );
	}
}

/*
========================
idPathTracerScene::AddLight

For PT_LIGHT_DIRECTIONAL, position is the normalized light direction.
========================
*/
void idPathTracerScene::AddLight( const idVec3 &position, const idVec3 &color, float intensity, int type,
								  float lightRadius, const float *lightProject, int projTexIndex, int falloffTexIndex,
								  const float *lightRadiusXYZ ) {
	if ( lights.Num() >= PT_MAX_LIGHTS ) {
		if ( r_ptDebug.GetBool() ) {
			common->Warning( "idPathTracerScene: light limit (%d) reached\n", PT_MAX_LIGHTS );
		}
		return;
	}

	ptLight_t light;
	memset( &light, 0, sizeof( light ) );
	light.type = type;
	if ( type == PT_LIGHT_DIRECTIONAL ) {
		light.direction[0] = position.x;
		light.direction[1] = position.y;
		light.direction[2] = position.z;
	} else {
		light.position[0] = position.x;
		light.position[1] = position.y;
		light.position[2] = position.z;
	}
	light.color[0] = color.x;
	light.color[1] = color.y;
	light.color[2] = color.z;
	light.intensity = intensity;
	light.radius = lightRadius;
	light.volumetric = r_ptVolumetric.GetFloat();
	light.textureIndex = -1;
	light.projectedTextureIndex = projTexIndex;
	light.falloffTextureIndex = falloffTexIndex;
	light.boxFalloff = ( type != PT_LIGHT_DIRECTIONAL && lightProject ) ? 1 : 0;

	for ( int i = 0; i < 3; i++ ) {
		light.lightRadius3[i] = lightRadiusXYZ ? lightRadiusXYZ[i] : lightRadius;
	}
	if ( lightProject ) {
		memcpy( light.lightProject, lightProject, sizeof( light.lightProject ) );
	}

	lights.Append( light );
}

/*
========================
idPathTracerScene::AddSpotLight
========================
*/
void idPathTracerScene::AddSpotLight( const idVec3 &position, const idVec3 &direction, const idVec3 &color,
									  float intensity, float coneAngle, float coneFalloff, float lightRadius,
									  const float *lightProject, int projTexIndex, int falloffTexIndex,
									  const float *rightAxis, const float *upAxis ) {
	if ( lights.Num() >= PT_MAX_LIGHTS ) {
		if ( r_ptDebug.GetBool() ) {
			common->Warning( "idPathTracerScene: light limit (%d) reached\n", PT_MAX_LIGHTS );
		}
		return;
	}

	ptLight_t light;
	memset( &light, 0, sizeof( light ) );
	light.type = PT_LIGHT_POINT;
	light.position[0] = position.x;
	light.position[1] = position.y;
	light.position[2] = position.z;
	light.direction[0] = direction.x;
	light.direction[1] = direction.y;
	light.direction[2] = direction.z;
	light.color[0] = color.x;
	light.color[1] = color.y;
	light.color[2] = color.z;
	light.intensity = intensity;
	light.radius = lightRadius;
	light.textureIndex = -1;
	light.volumetric = r_ptVolumetric.GetFloat() * 2.0f;
	light.coneAngle = coneAngle;
	light.coneFalloff = coneFalloff;
	light.projectedTextureIndex = projTexIndex;
	light.falloffTextureIndex = falloffTexIndex;
	light.boxFalloff = lightProject ? 1 : 0;
	light.lightRadius3[0] = light.lightRadius3[1] = light.lightRadius3[2] = lightRadius;

	if ( lightProject ) {
		memcpy( light.lightProject, lightProject, sizeof( light.lightProject ) );
	}
	if ( rightAxis ) {
		memcpy( light.right, rightAxis, sizeof( light.right ) );
	}
	if ( upAxis ) {
		memcpy( light.up, upAxis, sizeof( light.up ) );
	}

	lights.Append( light );
}

/*
========================
idPathTracerScene::EndFrame
========================
*/
void idPathTracerScene::EndFrame() {
	// area lights fill whatever the engine lights left over
	const int numArea = Min( areaLights.Num(), PT_MAX_LIGHTS - lights.Num() );
	for ( int i = 0; i < numArea; i++ ) {
		lights.Append( areaLights[i] );
	}
	// dropped area lights are only seen through their emission again
	for ( int i = numArea; i < areaLights.Num(); i++ ) {
		triangles[areaLightTris[i]].flags &= ~PT_TRI_AREA_LIGHT;
	}
	if ( numArea < areaLights.Num() && r_ptDebug.GetBool() ) {
		common->Printf( "idPathTracerScene: dropped %d emissive area lights (light limit)\n", areaLights.Num() - numArea );
	}

	bounds[0] = bounds[1] = bounds[2] = idMath::INFINITY;
	bounds[3] = bounds[4] = bounds[5] = -idMath::INFINITY;
	for ( int i = 0; i < triangles.Num(); i++ ) {
		const float *b = triangles[i].bounds;
		for ( int a = 0; a < 3; a++ ) {
			bounds[a]     = Min( bounds[a], b[a] );
			bounds[a + 3] = Max( bounds[a + 3], b[a + 3] );
		}
	}
	if ( triangles.Num() == 0 ) {
		memset( bounds, 0, sizeof( bounds ) );
	}

	if ( lights.Num() > 0 ) {
		contentHash = PT_HashWords( contentHash, lights.Ptr(), lights.Num() * sizeof( ptLight_t ) );
	}
	contentHash = PT_HashWords( contentHash, materials.Ptr(), materials.Num() * sizeof( ptMaterial_t ) );
	contentHash = PT_HashWords( contentHash, &textureGeneration, sizeof( textureGeneration ) );

	const int structure[5] = { triangles.Num(), lights.Num(), materials.Num(), textures.Num(), textureGeneration };
	structureHash = PT_HashWords( 2166136261u, structure, sizeof( structure ) );

	if ( r_ptDebug.GetBool() ) {
		common->Printf( "idPathTracerScene: %d verts, %d tris, %d materials, %d lights (%d area), %d textures (%d pending)\n",
			vertices.Num(), triangles.Num(), materials.Num(), lights.Num(), numArea, textures.Num(), pendingTextures.Num() );
	}
}

/*
========================
idPathTracerScene::AddTexture

Reads a GL texture back to system memory and queues it for the backend.
Returns the texture slot or -1 if the image can not be used.
========================
*/
int idPathTracerScene::AddTexture( const idImage *image ) {
	if ( !image || image->defaulted || image->texnum == idImage::TEXTURE_NOT_LOADED || image->type != TT_2D ) {
		return -1;
	}

	const int key = (int)(intptr_t)image;
	for ( int i = textureHash.First( key ); i >= 0; i = textureHash.Next( i ) ) {
		if ( texturePtrs[i] == image
				&& textureNameHashes[i] == PT_HashString( image->imgName.c_str() )
				&& textureGLNames[i] == image->texnum ) {
			return i;
		}
	}

	if ( textures.Num() >= PT_MAX_TEXTURES ) {
		if ( !needTextureFlush ) {
			common->Warning( "idPathTracerScene: texture limit (%d) reached, flushing next frame\n", PT_MAX_TEXTURES );
		}
		needTextureFlush = true;
		return -1;
	}

	GLint oldTexture;
	qglGetIntegerv( GL_TEXTURE_BINDING_2D, &oldTexture );
	qglBindTexture( GL_TEXTURE_2D, image->texnum );

	// the driver may have downscaled the image, trust GL over uploadWidth/Height
	GLint width = 0, height = 0;
	qglGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width );
	qglGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height );
	if ( width <= 0 || height <= 0 || width > PT_MAX_TEXTURE_SIZE || height > PT_MAX_TEXTURE_SIZE ) {
		qglBindTexture( GL_TEXTURE_2D, oldTexture );
		return -1;
	}

	ptTextureUpload_t upload;
	upload.width = width;
	upload.height = height;
	upload.rgba = new (std::nothrow) unsigned char[(size_t)width * height * 4];
	if ( !upload.rgba ) {
		qglBindTexture( GL_TEXTURE_2D, oldTexture );
		return -1;
	}
	qglGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, upload.rgba );
	qglBindTexture( GL_TEXTURE_2D, oldTexture );

	// GL returns luminance and intensity in the red channel only, expand them
	// to what the GL renderer samples (light and falloff images are mostly intensity)
	if ( image->internalFormat == GL_INTENSITY8 || image->internalFormat == GL_LUMINANCE8
		 || image->internalFormat == GL_LUMINANCE8_ALPHA8 ) {
		const bool intensity = image->internalFormat == GL_INTENSITY8;
		unsigned char *p = upload.rgba;
		for ( size_t i = 0, n = (size_t)width * height; i < n; i++, p += 4 ) {
			p[1] = p[2] = p[0];
			if ( intensity ) {
				p[3] = p[0];
			}
		}
	}

	ptTextureInfo_t info;
	info.width = width;
	info.height = height;

	const int index = textures.Append( info );
	texturePtrs.Append( image );
	textureNameHashes.Append( PT_HashString( image->imgName.c_str() ) );
	textureGLNames.Append( image->texnum );
	textureHash.Add( key, index );

	upload.index = index;
	pendingTextures.Append( upload );

	if ( r_ptDebug.GetBool() ) {
		common->Printf( "idPathTracerScene: texture %d: %s %dx%d\n", index, image->imgName.c_str(), width, height );
	}
	return index;
}
