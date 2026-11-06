#ifndef __PT_RENDER_H__
#define __PT_RENDER_H__

#include "renderer_pt/pt_backend.h"

/*
===============================================================================

	Path tracer front end, private to renderer_pt/. The engine facing entry
	points (RB_PT_Init, RB_PT_DrawView, ...) are declared in tr_local.h.

===============================================================================
*/

// the active backend, NULL until the first path traced frame
idPathTracerBackend *		RB_PT_GetBackend( void );

// true when the backend selected by r_ptBackend is compiled in and usable
bool						RB_PT_IsSupported( void );

// image pointers remain stable across reloadImages, so drop the PT readback cache
void						RB_PT_InvalidateImages( void );

#endif /* !__PT_RENDER_H__ */
