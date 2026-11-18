#ifndef __PT_MATERIAL_CACHE_H__
#define __PT_MATERIAL_CACHE_H__

#include <stdint.h>
#include "idlib/containers/List.h"
#include "idlib/containers/HashIndex.h"
#include "renderer_pt/pt_types.h"

// Keep a working set across frames, but recycle unused entries beyond this size.
static const int PT_MATERIAL_CACHE_REUSE_THRESHOLD = 1024;

class idPathTracerMaterialCache {
public:
	idPathTracerMaterialCache() { Clear(); }

	void Clear() {
		slots.Clear();
		hash.Free();
		const slot_t reserved = { NULL, 0, 0, -1, false };
		slots.Append( reserved );
		nextReclaim = PT_DEFAULT_MATERIAL + 1;
	}

	void BeginFrame() { nextReclaim = PT_DEFAULT_MATERIAL + 1; }
	int Num() const { return slots.Num(); }

	int Acquire( const void *material, unsigned int nameHash, bool animated,
				 unsigned int regHash, int frame, bool &rebuild ) {
		const int key = (int)(intptr_t)material;
		int reusable = -1;
		for ( int i = hash.First( key ); i >= 0; i = hash.Next( i ) ) {
			slot_t &slot = slots[i];
			if ( slot.material != material || slot.nameHash != nameHash ) {
				continue;
			}
			if ( slot.animated == animated && ( !animated || slot.regHash == regHash ) ) {
				slot.frame = frame;
				rebuild = false;
				return i;
			}
			if ( reusable < 0 && slot.frame != frame ) {
				reusable = i;
			}
		}

		if ( reusable < 0 && slots.Num() >= PT_MATERIAL_CACHE_REUSE_THRESHOLD ) {
			while ( nextReclaim < slots.Num() ) {
				const int candidate = nextReclaim++;
				if ( slots[candidate].frame != frame ) {
					reusable = candidate;
					break;
				}
			}
		}

		const slot_t slot = { material, nameHash, regHash, frame, animated };
		if ( reusable >= 0 ) {
			hash.Remove( (int)(intptr_t)slots[reusable].material, reusable );
			slots[reusable] = slot;
		} else {
			reusable = slots.Append( slot );
		}
		hash.Add( key, reusable );
		rebuild = true;
		return reusable;
	}

private:
	struct slot_t {
		const void *	material;
		unsigned int	nameHash;
		unsigned int	regHash;
		int				frame;
		bool			animated;
	};

	idList<slot_t>	slots;
	idHashIndex		hash;
	int				nextReclaim;
};

#endif /* !__PT_MATERIAL_CACHE_H__ */
