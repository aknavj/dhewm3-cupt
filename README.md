# DOOM3 GPU PATH TRACING RENDERER

This fork adds an experimental **GPU path tracer** (`neo/renderer_pt/`) that can replace the
classic OpenGL rasterizer for 3D views. The front end is compute-API agnostic; the only
backend implemented today is **CUDA** (`neo/renderer_pt/cuda/`). It is gated behind the
`CUDA_PATHTRACER` CMake option (off by default).

## Overview

Each frame the front end walks the view's draw surfaces and lights, builds a flat scene
(vertices, triangles, materials, lights, textures) on the CPU and hands it to the active
backend. The backend uploads it, builds a BVH on the GPU, path traces and tone maps the
image, and the front end stretches the result over the view's viewport. 2D views and
subviews (mirrors, remote cameras, portal skies) are still drawn with OpenGL, as is the HUD
and all GUI drawing on top of the 3D view.

## Source Layout

| Path | Contents |
|------|----------|
| `renderer_pt/pt_types.h` | POD types shared by host and device: `ptVertex_t`, `ptTriangle_t`, `ptMaterial_t`, `ptLight_t`, `ptBVHNode_t`, `ptRenderParams_t`, limits and enums |
| `renderer_pt/pt_backend.h` | `idPathTracerBackend` – abstract compute backend interface and factory type |
| `renderer_pt/pt_scene.h/.cpp`, `pt_scene_material.cpp` | `idPathTracerScene` – API-agnostic scene builder (geometry, Doom 3 material → PBR conversion, lights, texture readback, change hashes) |
| `renderer_pt/pt_render.h/.cpp` | Front end: cvars, backend registry / lifecycle, accumulation control, viewport blit (`RB_PT_*`) |
| `renderer_pt/pt_commands.cpp` | `r_ptCompare`, `r_ptSequence`, `r_ptAbort` capture tools |
| `renderer_pt/cuda/` | CUDA backend (`idPathTracerBackendCUDA`), kernels and the GPU LBVH builder |

## Adding a Backend (e.g. OpenCL)

1. Implement `idPathTracerBackend` (see `pt_backend.h`) in `neo/renderer_pt/<api>/`.
   The contract:
   - `Init` / `Resize` allocate the output for the maximum (window) size. `Render` may be called with any smaller trace size.
   - `UploadScene` receives an `idPathTracerScene`. Upload `PendingTextures()` into the slots given by their `index`. When `TextureGeneration()` changes, drop all cached textures.
   - `Render` accumulates samples into a float buffer until `ResetAccumulation()` is called, then tone maps it to RGBA8. Only `PT_MODE_PATHTRACE` accumulates; debug modes overwrite.
   - Report unrecoverable device errors through `IsHealthy()`, and the front end falls back to OpenGL.
   - All data types in `pt_types.h` are plain structs with no API-specific members, so they can be copied to the device as-is.
2. Export `PT_<API>_IsAvailable()` and `PT_<API>_CreateBackend()`, and add them to `ptBackendFactories[]` in `pt_render.cpp`.
3. In `neo/CMakeLists.txt`, add a detection block that defines `HAVE_PATHTRACER` and `HAVE_<API>`. Append your host sources to `src_renderer` and any device sources to `src_renderer_pt_device`.
4. Select it with `r_ptBackend <name>`. `auto` picks the first available backend.

## Rendering Modes (`r_ptMode`)

| Value | Mode |
|-------|------|
| 0 | **Path tracing** – Monte Carlo with Cook-Torrance BRDF, progressive accumulation |
| 1 | BVH debug – primary rays with random per-triangle colours |
| 2–6 | Texture debug – combined, base colour, albedo, normal map, specular map |
| 7 | **Whitted-style ray tracing** – direct lighting, Blinn-Phong specular, mirror reflections |

## Key Features

- **Linear BVH (LBVH)** built entirely on the GPU every frame using the Karras algorithm
  and a CUB radix sort, ordered on a single CUDA stream with no device-wide syncs
- **PBR materials** extracted from Doom 3 shader stages: albedo, normal map, specular,
  alpha test, blend modes, vertex colours, emission, glass (Fresnel + IOR). Time-varying shader
  registers are re-evaluated every frame, per entity
- **Stochastic direct lighting** with soft shadows, matching the GL light model: projected light
  image times falloff image, stage colours and condition registers, lights picked by estimated
  contribution. Shadow rays pass through alpha tested and translucent surfaces
- **sRGB correct shading**: textures and colours are decoded to linear before lighting
- **First person weapon** rendered as an overlay that never gets clipped by nearby walls
- **Indirect lighting** via cosine-weighted hemisphere sampling, plus emissive surfaces as area lights
- **Volumetric scattering** with a ray-marched Henyey-Greenstein phase function
- **Progressive accumulation** that resets on camera, mode, resolution or (configurable) scene
  changes, uses a running mean to preserve precision, and ramps quality over the first frames
- **Stackless BVH traversal** with escape links, so deep LBVH trees cannot overflow a per-ray stack
- **Cache invalidation** after `reloadImages`, plus identity checks to avoid stale cache hits when
  engine pointers are reused
- **Tone mapping**: Reinhard / ACES / Uncharted 2 with exposure and gamma
- **Comparison / sequence tools** for side-by-side GL vs. path-traced screenshots

## Building with CUDA

Requires the NVIDIA CUDA Toolkit and a host compiler it supports.

```
cmake -DCUDA_PATHTRACER=ON /path/to/repository/neo
```

By default, kernels are built for SM 75–90 (Turing → Hopper). Override this with
`-DCMAKE_CUDA_ARCHITECTURES=<list>`, for example `75` for a faster single-GPU build.

If your distribution's GCC is newer than nvcc supports (for example, GCC 16 with CUDA 13.1),
point nvcc at an older host compiler with `-DCMAKE_CUDA_HOST_COMPILER=g++-15`, or try:

```
CUDAFLAGS="-allow-unsupported-compiler -std=c++17" cmake -DCUDA_PATHTRACER=ON ...
```

Without `CUDA_PATHTRACER` the engine builds and runs as plain dhewm3, with no path tracer code compiled in.

## Console Commands

| Command | Description |
|---------|-------------|
| `r_ptInfo` | List compiled backends and print device / memory / timing info for the active one |
| `r_ptReset` | Restart accumulation and drop cached materials and textures |
| `r_ptCompare [samples]` | Save an OpenGL and a converged path-traced screenshot of the current view |
| `r_ptSequence <start> <end> [samples]` | Path trace a range of demo frames to screenshots |
| `r_ptAbort` | Abort `r_ptCompare` / `r_ptSequence` |
| `r_glSequence` / `r_glAbort` | OpenGL counterpart of `r_ptSequence` (always available) |

## Console Variables

| CVar | Default | Description |
|------|---------|-------------|
| `r_ptDraw` | `1` | Draw 3D views with the path tracer (0 = OpenGL) |
| `r_ptBackend` | `auto` | Compute backend: `auto`, `cuda` |
| `r_ptMode` | `0` | Render mode, see above |
| `r_ptDebug` | `0` | Print per-frame scene / timing info |
| `r_ptRenderScale` | `0.5` | Trace resolution relative to the viewport (0.1–1.0) |
| `r_ptSamplesPerPixel` | `4` | Samples per pixel per frame (1–64) |
| `r_ptMaxDepth` | `3` | Maximum ray bounces (1–16) |
| `r_ptMaxLightSamples` | `2` | Lights sampled per hit, picked by estimated contribution (1–8) |
| `r_ptIndirectProb` | `0.15` | Probability of the one bounce GI estimate, only used when `r_ptMaxDepth` is 1 (0–1) |
| `r_ptRussianRoulette` | `1` | Russian roulette path termination |
| `r_ptRRMinBounces` | `2` | Bounces before Russian roulette starts (0–16) |
| `r_ptRRSurvivalMin` | `0.1` | Minimum survival probability (0.01–0.95) |
| `r_ptEarlyTermThreshold` | `0.05` | Russian roulette for paths below this throughput after a bounce (0–0.5, unbiased) |
| `r_ptThroughputClamp` | `0.25` | Russian roulette for paths below this throughput after shading (0–1, unbiased) |
| `r_ptFireflyClamp` | `5.0` | Clamp per-sample radiance (0 = off) |
| `r_ptAccumulation` | `1` | Accumulate samples while the view is static |
| `r_ptResetOnSceneChange` | `1` | Restart accumulation on scene changes: 0 = camera only, 1 = objects/lights appearing or vanishing, 2 = any change incl. animation (may never converge) |
| `r_ptRayOffset` | `0.001` | Ray origin offset against self-intersection |
| `r_ptEmissionBoost` | `1.0` | Emission multiplier |
| `r_ptSpecularBoost` | `2.0` | Specular BRDF multiplier (Doom 3 uses 2×) |
| `r_ptSoftShadowScale` | `0.2` | Soft shadow jitter (0 = hard shadows) |
| `r_ptExposure` | `1.0` | Exposure multiplier |
| `r_ptGamma` | `2.2` | Output gamma, textures and colors are decoded with 2.2 |
| `r_ptToneMapMode` | `1` | 0 = Reinhard, 1 = ACES, 2 = Uncharted 2 |
| `r_ptSkyIntensity` | `0.0` | Sky dome intensity for primary ray misses |
| `r_ptSkyColorZenithR/G/B` | `0.3 / 0.35 / 0.5` | Sky zenith colour |
| `r_ptSkyColorHorizonR/G/B` | `0.4 / 0.35 / 0.3` | Sky horizon colour |
| `r_ptSkyColorGroundR/G/B` | `0.1 / 0.08 / 0.05` | Sky ground colour |
| `r_ptVolumetric` | `0.0` | Volumetric scattering intensity (0 = off, expensive) |
| `r_ptVolumetricDensity` | `0.008` | Fog density |
| `r_ptVolumetricSteps` | `16` | Ray-march steps (1–128) |
| `r_ptVolumetricAnisotropy` | `0.6` | Henyey-Greenstein g (−1 to 1) |
| `r_ptVolFalloff` | `2.0` | Distance falloff exponent |
| `r_ptVolMaxDist` | `500.0` | Maximum sampling distance |

The old `r_cu*` cvars have been renamed to `r_pt*`. Old config entries are ignored.

# ABOUT

_dhewm 3_ is a _Doom 3_ GPL source port, known to work on at least Windows, Linux, macOS and FreeBSD.

The goal of _dhewm 3_ is bring _DOOM 3_ with the help of SDL to all suitable
platforms.

Bugs present in the original _DOOM 3_ will be fixed (when identified) without
altering the original gameplay.

**The official homepage is:** https://dhewm3.org

**Mods supported by dhewm3:** https://dhewm3.org/mods.html

**Mod SDK for dhewm3:**  https://github.com/dhewm/dhewm3-sdk

**The project is hosted at:** https://github.com/dhewm

**Download the latest release:** https://github.com/dhewm/dhewm3/releases/latest

**Consult the FAQ at:** https://github.com/dhewm/dhewm3/wiki/FAQ

**Report bugs here:** https://github.com/dhewm/dhewm3/issues


# CHANGES

Compared to the original _DOOM 3_, the changes of _dhewm 3_ worth mentioning are:

- 64-bit port
- SDL for low-level OS support, OpenGL and input handling
- OpenAL for audio output, all OS-specific audio backends are gone
- OpenAL EFX for EAX reverb effects (read: EAX-like sound effects on all platforms/hardware)
- Gamepad support
- Better support for widescreen (and arbitrary display resolutions)
- A portable build system based on CMake
- (Cross-)compilation with MinGW-w64
- An advanced, mod-independent settings menu (opened with `F10` by default)

See [Changelog.md](./Changelog.md) for a more complete changelog.

# GENERAL NOTES

## Game data and patching

This source release does not contain any game data, the game data is still
covered by the original EULA and must be obeyed as usual.

You must patch the game to the latest version (1.3.1). See the FAQ for details, including
how to get the game data from Steam on Linux or OSX.

Note that the original _Doom 3_ and _Doom 3: Resurrection of Evil_ (together with
_DOOM 3: BFG Edition_, which is *not* supported by dhewm3) are available from the Steam Store at

https://store.steampowered.com/app/208200/DOOM_3/

See https://dhewm3.org/#how-to-install for game data installation instructions.

## Configuration

See [Configuration.md](./Configuration.md) for dhewm3-specific configuration, especially for 
using gamepads or the new settings menu.

## Compiling

The build system is based on CMake: http://cmake.org/

Required libraries are not part of the tree. These are:

- OpenAL (OpenAL Soft required, Creative's and Apple's versions are made of fail)
- SDL v1.2 or 2.0 (2.0 recommended)
- libcurl (optional, required for server downloads)
- Optionally, on non-Windows: libbacktrace (usually linked statically)
  - sometimes (e.g. on debian-based distros like Ubuntu) it's part of libgcc (=> always available),
    sometimes (e.g. Arch Linux, openSUSE) it's in a separate package
  - If this is available, dhewm3 prints more useful backtraces if it crashes

For **UNIX-like systems**, these libraries need to be installed (including the
developer files). It is recommended to use the software management tools of
your OS (apt, dnf, portage, BSD ports, [Homebrew for macOS](http://brew.sh), ...).

For **Windows**, there are three options:

- Use the provided binaries (recommended, see below)
- Compile these libraries yourself
- Use [vcpkg](https://vcpkg.io/) to install the dependencies
    - Remember to set `CMAKE_TOOLCHAIN_FILE` as described in their [Getting Started Guide](https://vcpkg.io/en/getting-started.html)

Create a distinct build folder outside of this source repository and issue
the cmake command there, pointing it at the neo/ folder from this repository:

`cmake /path/to/repository/neo`

**macOS** users need to point CMake at OpenAL Soft (better solutions welcome):

`cmake -DOPENAL_LIBRARY=/usr/local/opt/openal-soft/lib/libopenal.dylib -DOPENAL_INCLUDE_DIR=/usr/local/opt/openal-soft/include /path/to/repository/neo`

Newer versions of Homebrew install openal-soft to another directory, so use this instead:

`cmake -DOPENAL_LIBRARY="/opt/homebrew/opt/openal-soft/lib/libopenal.dylib" -DOPENAL_INCLUDE_DIR="/opt/homebrew/opt/openal-soft/include" /path/to/repo/neo`

### Compiling example using Ubuntu

Should be the same for Debian and other Debian-derivatives, but apart from the first step (installing
build dependecies) it should be the same on other Linux distros and even other UNIX-likes in general.

Open a terminal and follow these steps:

* Install build dependencies:  
  `sudo apt install git cmake build-essential libsdl2-dev libopenal-dev libcurl4-openssl-dev`  
    - The build-essential package on Debian/Ubuntu/... installs some basics for compiling code
      like GCC (incl. g++), GNU Make and the glibc development package
    - Instead of libcurl4-openssl-dev, other libcurl*-dev packages should also work - or none at all, curl is optional.
    - Not strictly necessary, but I recommend making libbacktrace available. On distributions not
      based on Debian, you may have to manually install some kind of libbacktrace development package.
* Use git to get the code from Github (alternatively you can also download the code as an archive and extract that):  
  `git clone https://github.com/dhewm/dhewm3.git`
* Change into the dhewm3 directory, create a directory to build in and change into the build directory:  
  `cd dhewm3` then `mkdir build` then `cd build`
* Create a Makefile with CMake: `cmake ../neo/`
    - You can set different options for CMake with arguments like `-DDEDICATED=ON` (to enable the dedicated server).
      You can show a list of supported options by running `cmake -LH ../neo/`. You can run CMake again
      with another `-DFOO=BAR` option to change that option (previously set options are remembered,
      unless you remove all contents of the build/ dir).
    - You could also install the **cmake-qt-gui** package and run `cmake-gui ../neo/`, which will let
      you configure the build in a GUI instead of using `-D` commandline-arguments.
* Compile dhewm3: `make -j8`
    - `-j8` specifies the number of compiler processes to run in parallel (8 in this example),
      it makes sense to use the number of CPU threads (or cores) in your system.

When all steps are done and no errors occurred, you should be able to run dhewm3 right there, like:  
`./dhewm3 +set fs_basepath /path/to/your/doom3/`  
*Replace `/path/to/your/doom3/` with the path to your Doom3 installation (that contains `base/` with
`pak000.pk4` to `pak008.pk4`)*

### Using the provided Windows binaries

Get a clone of the latest binaries here: https://github.com/dhewm/dhewm3-libs

There are two subfolders:

- 32-bit binaries are located in `i686-w64-mingw32`
- 64-bit binaries are located in `x86_64-w64-mingw32`

Issue the appropriate command from the build folder, for example (for VS2019 and 32bit):

`cmake -G "Visual Studio 16 2019" -A Win32 -DDHEWM3LIBS=/path/to/dhewm3-libs/i686-w64-mingw32 /path/to/repository/neo`

For 64bit dhewm3 binaries, use `-A x64` and `/path/to/dhewm3-libs/x86_64-w64-mingw32` instead (note that the official dhewm3 binaries for Windows are 32bit).  
For Visual Studio 2022 it's `"Visual Studio 17 2022"`.

For 32bit MinGW use:
`cmake -G "MinGW Makefiles" -DDHEWM3LIBS=/path/to/dhewm3-libs/i686-w64-mingw32 /path/to/repository/neo`

The binaries are compatible with MinGW-w64 and all MSVC versions.

### Cross-compiling

For cross-compiling, a CMake Toolchain file is required.

For the MinGW-w64 toolchain `i686-w64-mingw32` on Ubuntu 12.04, it looks like:

```
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR i686)

set(CMAKE_C_COMPILER i686-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER i686-w64-mingw32-g++)
set(CMAKE_RC_COMPILER i686-w64-mingw32-windres)

set(CMAKE_FIND_ROOT_PATH /usr/i686-w64-mingw32)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

Then point CMake at your toolchain file:
`cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/Toolchain.cmake -DDHEWM3LIBS=/path/to/dhewm3-libs/i686-w64-mingw32 /path/to/repository/neo`

**NOTE:** The `DHEWM3LIBS` path must be an **absolute path**, i.e. it must start with `/`!

If you want to build for x86_64 aka AMD64 aka x64, replace all instances of `i686`
in the toolchain file with `x86_64`.

## Back End Rendering of Stencil Shadows

The Doom 3 GPL source code release **did** not include functionality enabling rendering
of stencil shadows via the "depth fail" method, a functionality commonly known as
"Carmack's Reverse".  
It has been restored in dhewm3 1.5.1 after Creative Labs' [patent](https://patents.google.com/patent/US6384822B1/en)
finally expired.

Note that this did not change the visual appearance of the game, and didn't seem to
make a noticeable performance difference (on halfway-recent hardware) either.

## MayaImport

The code for our Maya export plugin is included, if you are a Maya licensee
you can obtain the SDK from Autodesk.


# LICENSE

See COPYING.txt for the GNU GENERAL PUBLIC LICENSE

ADDITIONAL TERMS:  The Doom 3 GPL Source Code is also subject to certain additional terms. You should have received a copy of these additional terms immediately following the terms and conditions of the GNU GPL which accompanied the Doom 3 Source Code.  If not, please request a copy in writing from id Software at id Software LLC, c/o ZeniMax Media Inc., Suite 120, Rockville, Maryland 20850 USA.

EXCLUDED CODE:  The code described below and contained in the Doom 3 GPL Source Code release is not part of the Program covered by the GPL and is expressly excluded from its terms.  You are solely responsible for obtaining from the copyright holder a license for such code and complying with the applicable license terms.

## Dear ImGui

neo/libs/imgui/*

The MIT License (MIT)

Copyright (c) 2014-2024 Omar Cornut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PropTree

neo/tools/common/PropTree/*

Copyright (C) 1998-2001 Scott Ramsay

sramsay@gonavi.com

http://www.gonavi.com

This material is provided "as is", with absolutely no warranty expressed
or implied. Any use is at your own risk.

Permission to use or copy this software for any purpose is hereby granted
without fee, provided the above notices are retained on all copies.
Permission to modify the code and to distribute modified code is granted,
provided the above notices are retained, and a notice that the code was
modified is included with the above copyright notice.

If you use this code, drop me an email.  I'd like to know if you find the code
useful.

## Base64 implementation

neo/idlib/Base64.cpp

Copyright (c) 1996 Lars Wirzenius.  All rights reserved.

June 14 2003: TTimo <ttimo@idsoftware.com>

modified + endian bug fixes

http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=197039

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## miniz

src/framework/miniz/*

The MIT License (MIT)

Copyright 2013-2014 RAD Game Tools and Valve Software
Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC

All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## IO on .zip files using minizip

src/framework/minizip/*

Copyright (C) 1998-2010 Gilles Vollant (minizip) ( http://www.winimage.com/zLibDll/minizip.html )

Modifications of Unzip for Zip64
Copyright (C) 2007-2008 Even Rouault

Modifications for Zip64 support
Copyright (C) 2009-2010 Mathias Svensson ( http://result42.com )

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

## MD4 Message-Digest Algorithm

neo/idlib/hashing/MD4.cpp

Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
rights reserved.

License to copy and use this software is granted provided that it
is identified as the "RSA Data Security, Inc. MD4 Message-Digest
Algorithm" in all material mentioning or referencing this software
or this function.

License is also granted to make and use derivative works provided
that such works are identified as "derived from the RSA Data
Security, Inc. MD4 Message-Digest Algorithm" in all material
mentioning or referencing the derived work.

RSA Data Security, Inc. makes no representations concerning either
the merchantability of this software or the suitability of this
software for any particular purpose. It is provided "as is"
without express or implied warranty of any kind.

These notices must be retained in any copies of any part of this
documentation and/or software.

## MD5 Message-Digest Algorithm

neo/idlib/hashing/MD5.cpp

This code implements the MD5 message-digest algorithm.
The algorithm is due to Ron Rivest.  This code was
written by Colin Plumb in 1993, no copyright is claimed.
This code is in the public domain; do with it what you wish.

## CRC32 Checksum

neo/idlib/hashing/CRC32.cpp

Copyright (C) 1995-1998 Mark Adler

## stb_image and stb_vorbis

neo/renderer/stb_image.h
neo/sound/stb_vorbis.h

Used to decode JPEG and OGG Vorbis files.

from https://github.com/nothings/stb/

Copyright (c) 2017 Sean Barrett

Released under MIT License and Unlicense (Public Domain)

## Brandelf utility

neo/sys/linux/setup/brandelf.c

Copyright (c) 1996 Søren Schmidt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer
   in this position and unchanged.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software withough specific prior written permission

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

`$FreeBSD: src/usr.bin/brandelf/brandelf.c,v 1.16 2000/07/02 03:34:08 imp Exp $`

## makeself - Make self-extractable archives on Unix

neo/sys/linux/setup/makeself/*, neo/sys/linux/setup/makeself/README
Copyright (c) Stéphane Peter
Licensing: GPL v2
