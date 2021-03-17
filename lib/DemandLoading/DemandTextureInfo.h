//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

namespace demandLoading {

const unsigned int MAX_TILE_LEVELS = 16;

/// Device-side texture info.
struct DemandTextureInfo
{
    // Texture dimensions
    unsigned int width;
    unsigned int height;

    // Filtering and initialization
    unsigned int mipLevels : 8;
    unsigned int mipTailFirstLevel : 8;
    unsigned int wrapMode0 : 2;
    unsigned int wrapMode1 : 2;
    unsigned int wrapMode2 : 2;
    unsigned int anisotropy : 8;
    unsigned int mipmapFilterMode : 1;
    unsigned int isInitialized : 1;

    // Tiling
    unsigned int tileWidth : 12;
    unsigned int tileHeight : 12;

    // Virtual addressing
    unsigned int startPage;
    unsigned int numPages;

    // Precomputed reciprocals
    float invWidth;
    float invHeight;
    float invTileWidth;
    float invTileHeight;
    float invAnisotropy;

    // Precomputed tile totals
    unsigned int numTilesBeforeLevel[MAX_TILE_LEVELS];
};

}  // namespace demandLoading
