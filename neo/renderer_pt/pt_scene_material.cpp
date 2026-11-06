#include "sys/platform.h"
#include "renderer/tr_local.h"
#include "renderer_pt/pt_scene.h"

/*
===============================================================================

	Material translation: idMaterial stages -> ptMaterial_t

===============================================================================
*/

extern idCVar r_ptDebug;

/*
========================
PT_InitMaterial
========================
*/
static void PT_InitMaterial( ptMaterial_t &mat ) {
    memset( &mat, 0, sizeof( mat ) );
    mat.albedo[0] = mat.albedo[1] = mat.albedo[2] = 0.5f;
    mat.albedo[3] = 1.0f;
    mat.roughness = 0.7f;
    mat.ior = 1.5f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;
    mat.emissionTexture = -1;
    mat.blendMode = PT_BLEND_OPAQUE;
    mat.bumpScale = 1.0f;
    mat.texTransform[0] = 1.0f;
    mat.texTransform[1] = 1.0f;
}

/*
========================
idPathTracerScene::AddDefaultMaterial

Slot PT_DEFAULT_MATERIAL is reserved for surfaces without a material and for
material table overflow, so those never alias a real material.
========================
*/
void idPathTracerScene::AddDefaultMaterial() {
    ptMaterial_t mat;
    PT_InitMaterial( mat );
    materials.Append( mat );
    materialEmission.Append( vec3_origin );
    materialPtrs.Append( NULL );
    materialFrame.Append( -1 );
    materialRegHash.Append( 0 );
}

/*
========================
idPathTracerScene::GetOrAddMaterial
========================
*/
int idPathTracerScene::GetOrAddMaterial( const idMaterial *material, const float *shaderRegisters ) {
    if ( !material ) {
        return PT_DEFAULT_MATERIAL;
    }

    // Animated materials (time / entity driven registers) are evaluated per
    // register set: entities sharing a material with different shaderParms get
    // separate slots. Slots are recycled across frames, a slot claimed this
    // frame is only shared by surfaces with identical registers.
    const bool animated = !material->ConstantRegisters() && shaderRegisters;
    const unsigned int regHash = animated
        ? PT_HashWords( 2166136261u, shaderRegisters, material->GetNumRegisters() * sizeof( float ) ) : 0;

    const int key = (int)(intptr_t)material;
    int firstSlot = -1;
    int freeSlot = -1;
    for ( int i = materialHash.First( key ); i >= 0; i = materialHash.Next( i ) ) {
        if ( materialPtrs[i] != material ) {
            continue;
        }
        if ( !animated ) {
            return i;
        }
        if ( firstSlot < 0 ) {
            firstSlot = i;
        }
        if ( materialRegHash[i] == regHash ) {
            materialFrame[i] = frameCount;
            return i;
        }
        if ( freeSlot < 0 && materialFrame[i] != frameCount ) {
            freeSlot = i;
        }
    }

    if ( freeSlot >= 0 ) {
        BuildMaterial( materials[freeSlot], materialEmission[freeSlot], material, shaderRegisters );
        materialFrame[freeSlot] = frameCount;
        materialRegHash[freeSlot] = regHash;
        return freeSlot;
    }

    if ( materials.Num() >= PT_MAX_MATERIALS ) {
        if ( firstSlot >= 0 ) {
            return firstSlot;	// share an instance rather than lose the material
        }
        common->Warning( "idPathTracerScene: material limit (%d) reached, '%s' uses the default material\n",
            PT_MAX_MATERIALS, material->GetName() );
        return PT_DEFAULT_MATERIAL;
    }

    const int index = materials.Num();
    ptMaterial_t mat;
    idVec3 emission;
    BuildMaterial( mat, emission, material, shaderRegisters );

    materials.Append( mat );
    materialEmission.Append( emission );
    materialPtrs.Append( material );
    materialFrame.Append( frameCount );
    materialRegHash.Append( regHash );
    materialHash.Add( key, index );

    if ( r_ptDebug.GetBool() ) {
        common->Printf( "idPathTracerScene: material %d: %s\n", index, material->GetName() );
    }
    return index;
}

/*
========================
idPathTracerScene::BuildMaterial
========================
*/
void idPathTracerScene::BuildMaterial( ptMaterial_t &mat, idVec3 &emission, const idMaterial *material, const float *shaderRegisters ) {
    PT_InitMaterial( mat );
    emission.Zero();

    if ( r_ptDebug.GetBool() ) {
        common->Printf( "BuildMaterial: %s (stages=%d, regs=%d, constRegs=%s)\n",
            material->GetName(), material->GetNumStages(), material->GetNumRegisters(),
            material->ConstantRegisters() ? "yes" : "no" );
    }

    // extract coverage type and alpha test threshold from engine material
    int coverage = 0; // local: 0=opaque, 1=perforated, 2=translucent
    materialCoverage_t cov = material->Coverage();
    if (cov == MC_PERFORATED) {
        coverage = 1;
        mat.alphaTest = 0.5f; // default threshold for perforated
    } else if (cov == MC_TRANSLUCENT) {
        coverage = 2;
        mat.alphaTest = 0.0f; // translucent uses continuous alpha
    }

    // extract noShadows flag from material (SurfaceCastsShadow checks MF_NOSHADOWS and MF_FORCESHADOWS)
    if (!material->SurfaceCastsShadow()) {
        mat.noShadows = 1;
    }

    // detect vertex color usage from stages (particles use per-vertex alpha for opacity)
    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
        if (stage && (stage->vertexColor == SVC_MODULATE || stage->vertexColor == SVC_INVERSE_MODULATE)) {
            mat.useVertexColor = 1;
            break;
        }
    }

    // check if material is ambient-only (no diffuse/bump/specular interaction stages)
    bool hasInteraction = false;
    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
        if (stage && (stage->lighting == SL_DIFFUSE || stage->lighting == SL_BUMP || stage->lighting == SL_SPECULAR)) {
            hasInteraction = true;
            break;
        }
    }
    if (!hasInteraction && material->GetNumStages() > 0) {
        mat.isAmbientOnly = 1;
    }

    // resolve registers: constant materials carry their own, animated ones use the
    // entity's evaluated registers; fall back to 1.0 when none are available
    const float *regs = material->ConstantRegisters();
    const int numRegs = material->GetNumRegisters();
    if ( !regs ) {
        regs = shaderRegisters;
    }

    // safe register lookup helper macro
    #define SAFE_REG(regIdx) ((regs && (regIdx) >= 0 && (regIdx) < numRegs) ? regs[regIdx] : 1.0f)

    // extract textures
    const shaderStage_t *ambientBaseStage = NULL;   // first non-additive ambient stage
    const shaderStage_t *ambientGlowStage = NULL;   // first additive ambient stage
    const shaderStage_t *ambientMaskStage = NULL;   // first maskcolor ambient stage (alpha-only write)
    const shaderStage_t *diffuseStage = NULL;
    const shaderStage_t *normalStage = NULL;
    const shaderStage_t *specularStage = NULL;

    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
		if (!stage) continue;

        // skip fragment program stages
        if (stage->newStage) {
            if (r_ptDebug.GetBool()) {
                common->Printf("  stage %d: SKIP (fragment program)\n", i);
            }
            continue;
        }

        // skip stages with non-standard texture coordinate generation
        if (stage->texture.texgen != TG_EXPLICIT) {
            if (r_ptDebug.GetBool()) {
                common->Printf("  stage %d: SKIP (texgen=%d)\n", i, (int)stage->texture.texgen);
            }
            continue;
        }

        switch (stage->lighting) {
            case SL_AMBIENT:
            {
                int blendBits = stage->drawStateBits & (GLS_SRCBLEND_BITS | GLS_DSTBLEND_BITS);
				if (blendBits == (GLS_SRCBLEND_ZERO | GLS_DSTBLEND_ONE)) {
					break;
				}

                // maskcolor stages only write alpha
                bool isMaskColor = ((stage->drawStateBits & GLS_COLORMASK) == GLS_COLORMASK);
                if (isMaskColor) {
                    if (!ambientMaskStage) ambientMaskStage = stage;
                    if (r_ptDebug.GetBool()) {
                        common->Printf("  stage %d: maskcolor (alpha mask)\n", i);
                    }
                    break;
                }

                bool stageIsAdd = (blendBits == (GLS_SRCBLEND_ONE | GLS_DSTBLEND_ONE));
                if (stageIsAdd) {
                    if (!ambientGlowStage) ambientGlowStage = stage;
                } else {
                    if (!ambientBaseStage) ambientBaseStage = stage;
                }
                break;
            }
            case SL_DIFFUSE:
            {
                int blendBits = stage->drawStateBits & (GLS_SRCBLEND_BITS | GLS_DSTBLEND_BITS);
				if (blendBits == (GLS_SRCBLEND_ZERO | GLS_DSTBLEND_ONE)) {
					break;
				}
                diffuseStage = stage;
                break;
            }
            case SL_BUMP:
            {
                normalStage = stage;
                break;
            }
            case SL_SPECULAR:
            {
                specularStage = stage;
                break;
            }
            default:
                break;
        }
    }

    bool hasDiffuse = (diffuseStage != NULL);

    // validate perforated coverage
    if (coverage == 1) {
        bool usableAlphaTest = false;
        if (diffuseStage && diffuseStage->hasAlphaTest) usableAlphaTest = true;
        if (ambientBaseStage && ambientBaseStage->hasAlphaTest) usableAlphaTest = true;
        if (ambientGlowStage && ambientGlowStage->hasAlphaTest) usableAlphaTest = true;
        if (!usableAlphaTest) {
            if (r_ptDebug.GetBool()) {
                common->Printf("  coverage=1 but no usable alpha-tested stage, downgrading to opaque\n");
            }
            coverage = 0;
            mat.alphaTest = 0.0f;
        }
    }

    if (r_ptDebug.GetBool()) {
        common->Printf("  stages: coverage=%d ambBase=%s ambGlow=%s ambMask=%s diff=%s norm=%s spec=%s\n",
            coverage,
            ambientBaseStage ? "yes" : "no",
            ambientGlowStage ? "yes" : "no",
            ambientMaskStage ? "yes" : "no",
            hasDiffuse ? "yes" : "no",
            normalStage ? "yes" : "no",
            specularStage ? "yes" : "no");
    }

    if (hasDiffuse) {
        mat.albedoTexture = AddTexture(diffuseStage->texture.image);
        if (mat.albedoTexture >= 0) {
            mat.albedo[0] = SAFE_REG(diffuseStage->color.registers[0]);
            mat.albedo[1] = SAFE_REG(diffuseStage->color.registers[1]);
            mat.albedo[2] = SAFE_REG(diffuseStage->color.registers[2]);
        }

        // refine alpha test threshold from stage draw state bits
        if (coverage == 1 || coverage == 2) {
            int atest = diffuseStage->drawStateBits & GLS_ATEST_BITS;
            if (atest & GLS_ATEST_GE_128) {
                mat.alphaTest = 128.0f / 255.0f; // ~0.502
            } else if (atest & GLS_ATEST_LT_128) {
                mat.alphaTest = 128.0f / 255.0f;
            } else if (atest & GLS_ATEST_EQ_255) {
                mat.alphaTest = 254.0f / 255.0f;
            }
        }

        if (r_ptDebug.GetBool()) {
            common->Printf("  %s: coverage=%d alphaTest=%.3f\n", material->GetName(), coverage, mat.alphaTest);
        }
    }

    // process glow stage (emission)
    int emissionTextureIdx = -1;
    if (ambientGlowStage && ambientGlowStage->texture.image && !mat.isAmbientOnly) {
        int emTex = AddTexture(ambientGlowStage->texture.image);
        if (emTex >= 0) {
            emissionTextureIdx = emTex;
            mat.emissionTexture = emTex;
            float emR = SAFE_REG(ambientGlowStage->color.registers[0]);
            float emG = SAFE_REG(ambientGlowStage->color.registers[1]);
            float emB = SAFE_REG(ambientGlowStage->color.registers[2]);
            mat.emission[0] = emR;
            mat.emission[1] = emG;
            mat.emission[2] = emB;
            emission.Set(mat.emission[0], mat.emission[1], mat.emission[2]);
        }
    }

    // for materials with NO diffuse stage
    if (!hasDiffuse && mat.albedoTexture < 0) {
        const shaderStage_t *albedoSource = ambientBaseStage;
        if (!albedoSource) albedoSource = ambientMaskStage;
        if (!albedoSource) albedoSource = ambientGlowStage;

        if (albedoSource && albedoSource->texture.image) {
            int ambTex = AddTexture(albedoSource->texture.image);
            if (ambTex >= 0) {
                mat.albedoTexture = ambTex;
                mat.albedo[0] = SAFE_REG(albedoSource->color.registers[0]);
                mat.albedo[1] = SAFE_REG(albedoSource->color.registers[1]);
                mat.albedo[2] = SAFE_REG(albedoSource->color.registers[2]);
                mat.albedo[3] = idMath::ClampFloat(0.0f, 1.0f, SAFE_REG(albedoSource->color.registers[3]));

                // extract blend mode from ambient stage draw state bits
                int srcBlend = albedoSource->drawStateBits & GLS_SRCBLEND_BITS;
                int dstBlend = albedoSource->drawStateBits & GLS_DSTBLEND_BITS;

                if (srcBlend == GLS_SRCBLEND_ONE && dstBlend == GLS_DSTBLEND_ONE) {
                    mat.blendMode = PT_BLEND_ADD;
                } else if (srcBlend == GLS_SRCBLEND_SRC_ALPHA && dstBlend == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA) {
                    mat.blendMode = PT_BLEND_ALPHA;
                } else if ((srcBlend == GLS_SRCBLEND_DST_COLOR && dstBlend == GLS_DSTBLEND_ZERO)
                        || (srcBlend == GLS_SRCBLEND_ZERO && dstBlend == GLS_DSTBLEND_SRC_COLOR)) {
                    mat.blendMode = PT_BLEND_MODULATE;
                } else if (srcBlend != GLS_SRCBLEND_ONE || dstBlend != GLS_DSTBLEND_ZERO) {
                    // Non-opaque blend mode: check for additive variant (e.g. GL_SRC_ALPHA + GL_ONE)
                    if (dstBlend == GLS_DSTBLEND_ONE) {
                        mat.blendMode = PT_BLEND_ADD;
                    } else {
                        mat.blendMode = PT_BLEND_ALPHA;
                    }
                }

                // ambient-only surfaces without a glow map get mild self-emission
                if (mat.blendMode == 0 && emissionTextureIdx < 0) {
                    mat.emission[0] = mat.albedo[0] * 0.3f;
                    mat.emission[1] = mat.albedo[1] * 0.3f;
                    mat.emission[2] = mat.albedo[2] * 0.3f;
                    emission.Set(mat.emission[0], mat.emission[1], mat.emission[2]);
                }
            }
        }
    }

    // classify blend modes for translucent materials
    {
        const shaderStage_t *blendStage = ambientBaseStage ? ambientBaseStage : ambientGlowStage;
        if (blendStage && coverage == 2) {
            int ambSrc = blendStage->drawStateBits & GLS_SRCBLEND_BITS;
            int ambDst = blendStage->drawStateBits & GLS_DSTBLEND_BITS;

            bool isAdditive = (ambSrc == GLS_SRCBLEND_ONE && ambDst == GLS_DSTBLEND_ONE);
            bool isAlphaBlend = (ambSrc == GLS_SRCBLEND_SRC_ALPHA && ambDst == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA);
            bool isPremultAlpha = (ambSrc == GLS_SRCBLEND_ONE && ambDst == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA);
            bool isFilter = (ambSrc == GLS_SRCBLEND_DST_COLOR && ambDst == GLS_DSTBLEND_ZERO)
                         || (ambSrc == GLS_SRCBLEND_ZERO && ambDst == GLS_DSTBLEND_SRC_COLOR);

            if (isAdditive) {
                mat.blendMode = PT_BLEND_ADD;
            } else if (isAlphaBlend || isPremultAlpha) {
                mat.blendMode = PT_BLEND_ALPHA;
            } else if (isFilter) {
                mat.blendMode = PT_BLEND_MODULATE;
            }

            // for blended surfaces, use the blend stage texture as albedo if we don't have one
            if (mat.blendMode > 0 && blendStage->texture.image && mat.albedoTexture < 0) {
                mat.albedoTexture = AddTexture(blendStage->texture.image);
                mat.albedo[0] = SAFE_REG(blendStage->color.registers[0]);
                mat.albedo[1] = SAFE_REG(blendStage->color.registers[1]);
                mat.albedo[2] = SAFE_REG(blendStage->color.registers[2]);
                mat.albedo[3] = SAFE_REG(blendStage->color.registers[3]);
            }

            if (r_ptDebug.GetBool()) {
                common->Printf("  %s: blendMode=%d\n", material->GetName(), mat.blendMode);
            }
        }
    }

    // process normal map stage
    bool hasNormal = (normalStage != NULL);
    if (hasNormal) {
        mat.normalTexture = AddTexture(normalStage->texture.image);
        if (r_ptDebug.GetBool()) {
            common->Printf("  %s: normal map %d\n", material->GetName(), mat.normalTexture);
        }
    }

    // process specular map stage
    bool hasSpecular = (specularStage != NULL);
    if (hasSpecular) {
        mat.specularTexture = AddTexture(specularStage->texture.image);
        if (r_ptDebug.GetBool()) {
            common->Printf("  %s: specular map %d\n", material->GetName(), mat.specularTexture);
        }
    }

    // colors are multiplied in gamma space by the GL renderer, decode every
    // factor so that the product encoded with r_ptGamma 2.2 matches it
    for ( int c = 0; c < 3; c++ ) {
        mat.albedo[c] = PT_SrgbToLinear( mat.albedo[c] );
        mat.emission[c] = PT_SrgbToLinear( mat.emission[c] );
    }
    emission.Set( mat.emission[0], mat.emission[1], mat.emission[2] );

    #undef SAFE_REG
}
