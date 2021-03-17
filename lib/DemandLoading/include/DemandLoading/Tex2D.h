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

#include <DemandLoading/DemandTextureContext.h>
#include <DemandLoading/DemandTextureInfo.h>
#include <DemandLoading/TileIndexing.h>

#include <optixPaging/optixPaging.h>

#include <sutil/vec_math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace demandLoading {

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )

// Compute tile level from the texture gradients
inline __device__ float getMipLevelFromTextureGradients( float2 ddx, float2 ddy, int texWidth, int texHeight, float invAnisotropy )
{
    const float2 scale = make_float2( texWidth, texHeight );
    ddx *= scale;
    ddy *= scale;

    // Trying to follow CUDA.
    // Our experiments suggest that CUDA performs a low precision EWA filter
    // correction on the texture gradients to determine the mip level.
    // This calculation is described in the Siggraph 1999 paper:
    // Feline: Fast Elliptical Lines for Anisotropic Texture Mapping

    const float A    = ddy.x * ddy.x + ddy.y * ddy.y;
    const float B    = -2.0f * ( ddx.x * ddy.x + ddx.y * ddy.y );
    const float C    = ddx.x * ddx.x + ddx.y * ddx.y;
    const float root = sqrt( maximum( A * A - 2.0f * A * C + C * C + B * B, 0.0f ) );

    // Compute the square of the major and minor ellipse radius lengths to avoid sqrts.
    // Then compensate by taking half the log to get the tile level.

    const float minorRadius2 = ( A + C - root ) * 0.5f;
    const float majorRadius2 = ( A + C + root ) * 0.5f;
    const float filterWidth2 = maximum( minorRadius2, majorRadius2 * invAnisotropy * invAnisotropy );
    const float mipLevel     = 0.5f * log2f( filterWidth2 );
    return mipLevel;
}

template <class TYPE>
__inline__ __device__ TYPE
tex2DGrad( const DemandTextureContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    const DemandTextureInfo&  info = context.m_textureInfos[textureId];
    const OptixPagingContext& op   = context.m_pagingContext;

    // If the textureInfo has not been initialized, request the start page and return a dummy value
    if( !info.isInitialized )
    {
        optixPagingMapOrRequest( op.usageBits, op.residenceBits, op.pageTable, info.startPage, isResident );
        *isResident = false;
        return TYPE();
    }

    float mipLevel = getMipLevelFromTextureGradients( ddx, ddy, info.width, info.height, info.invAnisotropy );
    // Note that if the texture is not mipmapped the mipLevel will be clamped to 0
    mipLevel = clamp( mipLevel, 0.0f, info.mipLevels - 1.0f );

    // Snap to the nearest mip level if we're doing point filtering
    if( info.mipmapFilterMode == cudaFilterModePoint )
        mipLevel = floorf( mipLevel + 0.5f );

    const unsigned int coarseLevel = static_cast<unsigned int>( ceilf( mipLevel ) );

    // Request up to 4 tiles if the sample is near the border of a tile
    bool         resident = true;
    bool         tileResident;
    unsigned int numTileRequests = 0;
    unsigned int tileRequests[4];
    calculateTileRequests( info, coarseLevel, x, y, tileRequests, numTileRequests );
    for( int i = 0; i < numTileRequests; ++i )
    {
        optixPagingMapOrRequest( op.usageBits, op.residenceBits, op.pageTable, info.startPage + tileRequests[i], &tileResident );
        resident &= tileResident;
    }
    *isResident = resident;

    const unsigned int fineLevel = static_cast<unsigned int>( floorf( mipLevel ) );
    if( fineLevel != coarseLevel )
    {
        // Request up to 4 tiles if the sample is near the border of a tile
        calculateTileRequests( info, fineLevel, x, y, tileRequests, numTileRequests );
        for( int i = 0; i < numTileRequests; ++i )
        {
            optixPagingMapOrRequest( op.usageBits, op.residenceBits, op.pageTable, info.startPage + tileRequests[i], &tileResident );
            resident &= tileResident;
        }
        *isResident = *isResident && resident;
    }

    if( !*isResident )
        return TYPE();

    const cudaTextureObject_t& texture = context.m_textures[textureId];
    return tex2DGrad<TYPE>( texture, x, y, ddx, ddy );
}

// Do isotropic sample of a demand loaded tiled texture based on a mip level
template <class TYPE>
__inline__ __device__ TYPE tex2DLod( const DemandTextureContext& context, unsigned int textureId, float s, float t, float mipLevel, bool* isResident )
{
    const DemandTextureInfo& info = context.m_textureInfos[textureId];

    float sampleWidth  = 1.0f;
    float sampleHeight = 1.0f;
    if( info.isInitialized )
    {
        const float expMipLevel = exp2( mipLevel );
        sampleWidth             = expMipLevel * info.invWidth;
        sampleHeight            = expMipLevel * info.invHeight;
    }
    const float2 ddx = make_float2( sampleWidth, 0.0f );
    const float2 ddy = make_float2( 0.0f, sampleHeight );
    return tex2DGrad<TYPE>( context, textureId, s, t, ddx, ddy, isResident );
}

#endif

}  // namespace demandLoading
