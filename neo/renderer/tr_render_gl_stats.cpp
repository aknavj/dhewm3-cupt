#include "sys/platform.h"
#include "framework/Console.h"
#include "renderer/tr_local.h"

/*
========================
OpenGL Sequence Renderer
========================
*/
enum glSequenceState_t {
	GL_SEQ_IDLE = 0,
	GL_SEQ_SETUP,
	GL_SEQ_WAIT_RENDER,
	GL_SEQ_SCREENSHOT,
	GL_SEQ_ADVANCE_UNPAUSE,
	GL_SEQ_ADVANCE_WAIT,
	GL_SEQ_ADVANCE_REFREEZE,
	GL_SEQ_DONE
};

static glSequenceState_t glSeq_state = GL_SEQ_IDLE;
static int glSeq_frameStart = 0;
static int glSeq_frameEnd = 0;
static int glSeq_currentFrame = 0;
static int glSeq_waitCounter = 0;
static int glSeq_startTime = 0;
static int glSeq_saved_fixedTic = 0;
static bool glSeq_saved_stopTime = false;

/*
========================
R_GLSequence_f
========================
*/
void R_GLSequence_f( const idCmdArgs &args ) {
	if (glSeq_state != GL_SEQ_IDLE) {
		common->Printf("R_GLSequence_f(): Sequence render already in progress. Use r_glAbort to cancel.\n");
		return;
	}

	if (args.Argc() < 3) {
		common->Printf("Usage: r_glSequence <frame_start> <frame_end>\n");
		common->Printf("  frame_start  - First game frame to render (0+)\n");
		common->Printf("  frame_end    - Last game frame to render (inclusive)\n");
		common->Printf("\nThe game will be paused and single-stepped forward between frames.\n");
		common->Printf("Output: screenshots/gl_screenshot_frame<N>.<ext>\n");
		return;
	}

	glSeq_frameStart = atoi(args.Argv(1));
	glSeq_frameEnd = atoi(args.Argv(2));
	if (glSeq_frameStart < 0) glSeq_frameStart = 0;
	if (glSeq_frameEnd < glSeq_frameStart) {
		common->Printf("R_GLSequence_f(): Error: frame_end (%d) must be >= frame_start (%d)\n",
			glSeq_frameEnd, glSeq_frameStart);
		return;
	}

	int totalFrames = glSeq_frameEnd - glSeq_frameStart + 1;

	common->Printf("\n=== OpenGL Sequence Render ===\n");
	common->Printf("  Frames:  %d - %d (%d total)\n", glSeq_frameStart, glSeq_frameEnd, totalFrames);
	common->Printf("  Output:  screenshots/gl_screenshot_frame<N>.<ext>\n");
	common->Printf("==============================\n\n");

	// close console so it doesn't appear in screenshots
	console->Close();

	glSeq_currentFrame = 0;
	glSeq_state = GL_SEQ_SETUP;
}

/*
========================
R_GLAbort_f
========================
*/
void R_GLAbort_f( const idCmdArgs &args ) {
	if (glSeq_state == GL_SEQ_IDLE) {
		common->Printf("R_GLAbort_f(): Nothing in progress to abort.\n");
		return;
	}

	common->Printf("R_GLAbort_f(): Aborting sequence render at frame %d (was in state %d)...\n",
		glSeq_currentFrame, (int)glSeq_state);

	// restore saved cvar values
	idCVar* stopTimeCVar = cvarSystem->Find("g_stoptime");
	if (stopTimeCVar) {
		stopTimeCVar->SetBool(glSeq_saved_stopTime);
	}
	int restoreFixedTic = glSeq_saved_fixedTic;
	if (restoreFixedTic == 1) restoreFixedTic = 0;
	cvarSystem->SetCVarInteger("com_fixedTic", restoreFixedTic);

	glSeq_state = GL_SEQ_IDLE;
	common->Printf("R_GLAbort_f(): Aborted. CVars restored.\n");
}

/*
========================
RB_GLSequenceCheck
========================
*/
void RB_GLSequenceCheck( void ) {

	if (glSeq_state == GL_SEQ_IDLE) {
		return;
	}

	switch (glSeq_state) {

        case GL_SEQ_SETUP: 
        {
            // save current cvar values
            idCVar* stopTimeCVar = cvarSystem->Find("g_stoptime");
            glSeq_saved_stopTime = stopTimeCVar ? stopTimeCVar->GetBool() : false;
            glSeq_saved_fixedTic = cvarSystem->GetCVarInteger("com_fixedTic");

            // set com_fixedTic 1 for deterministic single-tick stepping
            cvarSystem->SetCVarInteger("com_fixedTic", 1);

            glSeq_startTime = Sys_Milliseconds();

            // if frame_start > 0, we need to advance that many ticks first.
            if (glSeq_frameStart > 0) {
                if (stopTimeCVar) {
                    stopTimeCVar->SetBool(false);
                }
                glSeq_currentFrame = 0;
                glSeq_waitCounter = 0;
                glSeq_state = GL_SEQ_ADVANCE_UNPAUSE;
                common->Printf("RB_GLSequenceCheck(): Advancing to frame %d...\n", glSeq_frameStart);
            } else {
                // start from frame 0 — freeze immediately
                if (stopTimeCVar) {
                    stopTimeCVar->SetBool(true);
                }
                glSeq_currentFrame = 0;
                glSeq_waitCounter = 0;
                glSeq_state = GL_SEQ_WAIT_RENDER;
                common->Printf("RB_GLSequenceCheck(): Frame %d / %d: rendering...\n",
                    glSeq_currentFrame, glSeq_frameEnd);
            }
            break;
        }

        case GL_SEQ_WAIT_RENDER: 
        {
            // wait one frame for the GL renderer to produce a stable image
            glSeq_waitCounter++;
            if (glSeq_waitCounter >= 2) {
                glSeq_state = GL_SEQ_SCREENSHOT;
            }
            break;
        }

        case GL_SEQ_SCREENSHOT: 
        {
            // determine screenshot format
            int format = cvarSystem->GetCVarInteger("r_screenshotFormat");
            const char* ext = "tga";
            switch (format) {
                case 1: ext = "bmp"; break;
                case 2: ext = "png"; break;
                case 3: ext = "jpg"; break;
                default: ext = "tga"; break;
            }

            idStr filename = va("screenshots/gl_screenshot_frame%05d.%s", glSeq_currentFrame, ext);
            tr.TakeScreenshot(glConfig.vidWidth, glConfig.vidHeight, filename.c_str(), 1, NULL);

            int totalFrames = glSeq_frameEnd - glSeq_frameStart + 1;
            int framesDone = glSeq_currentFrame - glSeq_frameStart + 1;
            float elapsedSec = (float)(Sys_Milliseconds() - glSeq_startTime) * 0.001f;

            common->Printf("frame: %d | time: %.2fs | saved: %s (%d/%d, %d%%)\n",
                glSeq_currentFrame, elapsedSec, filename.c_str(), framesDone, totalFrames,
                (framesDone * 100) / totalFrames);

            // check if this was the last frame
            if (glSeq_currentFrame >= glSeq_frameEnd) {
                glSeq_state = GL_SEQ_DONE;
            } else {
                glSeq_state = GL_SEQ_ADVANCE_UNPAUSE;
            }
            break;
        }

        case GL_SEQ_ADVANCE_UNPAUSE: 
        {
            // unpause the game so the next RunGameTic executes
            idCVar* stopTimeCVar = cvarSystem->Find("g_stoptime");
            if (stopTimeCVar) {
                stopTimeCVar->SetBool(false);
            }
            glSeq_waitCounter = 0;
            glSeq_state = GL_SEQ_ADVANCE_WAIT;
            break;
        }

        case GL_SEQ_ADVANCE_WAIT: 
        {
            // wait one frame for the game tick to actually execute
            glSeq_waitCounter++;
            if (glSeq_waitCounter >= 1) {
                glSeq_currentFrame++;
                glSeq_state = GL_SEQ_ADVANCE_REFREEZE;
            }
            break;
        }

        case GL_SEQ_ADVANCE_REFREEZE: 
        {
            // re-freeze the game
            idCVar* stopTimeCVar = cvarSystem->Find("g_stoptime");
            if (stopTimeCVar) {
                stopTimeCVar->SetBool(true);
            }

            // if we haven't reached the start frame yet, keep advancing
            if (glSeq_currentFrame < glSeq_frameStart) {
                if (glSeq_currentFrame % 100 == 0) {
                    common->Printf("RB_GLSequenceCheck(): Fast-forward: frame %d / %d...\n",
                        glSeq_currentFrame, glSeq_frameStart);
                }
                glSeq_state = GL_SEQ_ADVANCE_UNPAUSE;
            } else {
                // renderable frame — wait for GL to render then screenshot
                glSeq_waitCounter = 0;
                glSeq_state = GL_SEQ_WAIT_RENDER;
                common->Printf("RB_GLSequenceCheck(): Frame %d / %d: rendering...\n",
                    glSeq_currentFrame, glSeq_frameEnd);
            }
            break;
        }

        case GL_SEQ_DONE: 
        {
            // restore saved cvar values
            idCVar* stopTimeCVar = cvarSystem->Find("g_stoptime");
            if (stopTimeCVar) {
                stopTimeCVar->SetBool(glSeq_saved_stopTime);
            }

            int restoreFixedTic = glSeq_saved_fixedTic;
            if (restoreFixedTic == 1) {
                common->Printf("RB_GLSequenceCheck(): Saved com_fixedTic was %d (sequence value), forcing to 0\n", restoreFixedTic);
                restoreFixedTic = 0;
            }
            cvarSystem->SetCVarInteger("com_fixedTic", restoreFixedTic);

            common->Printf("RB_GLSequenceCheck(): Restored: g_stopTime=%d, com_fixedTic=%d\n",
                (int)glSeq_saved_stopTime, restoreFixedTic);

            int totalFrames = glSeq_frameEnd - glSeq_frameStart + 1;
            float totalSec = (float)(Sys_Milliseconds() - glSeq_startTime) * 0.001f;
            common->Printf("\n=== OpenGL Sequence Render Complete ===\n");
            common->Printf("  Rendered %d frames (%d - %d) in %.1fs\n",
                totalFrames, glSeq_frameStart, glSeq_frameEnd, totalSec);
            common->Printf("  Output: screenshots/gl_screenshot_frame%05d - gl_screenshot_frame%05d\n",
                glSeq_frameStart, glSeq_frameEnd);
            common->Printf("=========================================\n\n");

            glSeq_state = GL_SEQ_IDLE;
            break;
        }

        case GL_SEQ_IDLE:
        {
            break;
        }

        default:
        {
            break;
        }
    }
}
