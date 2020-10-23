//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <DemandLoading/DemandTextureInfo.h>

#include <cuda_runtime.h>
#include <texture_types.h>

#ifndef __CUDACC__
#include <algorithm>
#include <cmath>
#endif

namespace demandLoading {

#ifdef __CUDACC__
#define HOSTDEVICE __device__
#else
#define HOSTDEVICE __host__
#endif

// clang-format off
#ifdef __CUDACC__
HOSTDEVICE inline int ifloor( float x ) { return static_cast<int>( ::floorf( x ) ); }
HOSTDEVICE inline float floorf( float x ) { return ::floorf( x ); }
HOSTDEVICE inline float ceilf( float x ) { return ::ceilf( x ); }
HOSTDEVICE inline float maxf( float x, float y ) { return ::fmaxf( x, y ); }
HOSTDEVICE inline float minf( float x, float y ) { return ::fminf( x, y ); }
HOSTDEVICE inline unsigned int uimax( unsigned int a, unsigned int b ) { return ( a > b ) ? a : b; }
#else
HOSTDEVICE inline int ifloor( float x ) { return static_cast<int>( std::floor( x ) ); }
HOSTDEVICE inline float floorf( float x ) { return std::floor( x ); }
HOSTDEVICE inline float ceilf( float x ) { return std::ceil( x ); }
HOSTDEVICE inline float maxf( float x, float y ) { return std::max( x, y ); }
HOSTDEVICE inline float minf( float x, float y ) { return std::min( x, y ); }
HOSTDEVICE inline unsigned int uimax( unsigned int a, unsigned int b ) { return std::max( a, b ); }
#endif
HOSTDEVICE inline float clampf( float f, float a, float b ) { return maxf( a, minf( f, b ) ); }
// clang-format on

HOSTDEVICE inline unsigned int ceilMult( unsigned int num, unsigned int den, float invDen )
{
    // This should work as long as (num + dem - 1) < 16M
    return static_cast<unsigned int>( static_cast<float>( num + den - 1 ) * invDen );
}

HOSTDEVICE inline unsigned int calculateLevelDim( unsigned int mipLevel, unsigned int textureDim )
{
    return uimax( textureDim >> mipLevel, 1U );
}

HOSTDEVICE inline unsigned int wrapPixelCoord( cudaTextureAddressMode addressMode, int coord, int max )
{
    if( addressMode == cudaAddressModeClamp || addressMode == cudaAddressModeBorder )
        return coord < 0 ? 0 : ( coord >= max ? max - 1 : coord );

    // wrap and mirror modes
    // Compute (floored) quotient and remainder
    const int q = ifloor( static_cast<float>( coord ) / static_cast<float>( max ) );
    const int r = coord - q * max;
    // In mirror mode, flip the coordinate (r) if the q is odd
    return ( addressMode == cudaAddressModeMirror && ( q & 1 ) ) ? ( max - 1 - r ) : r;
}

HOSTDEVICE inline float wrapNormCoord( cudaTextureAddressMode addressMode, float x )
{
    const float firstFloatLessThanOne = 0.999999940395355224609375f;

    if( addressMode == cudaAddressModeClamp || addressMode == cudaAddressModeBorder )
        return clampf( x, 0.0f, firstFloatLessThanOne );  // result must be < 1

    // Wrap and mirror modes
    const int xfloor = ifloor( x );
    if( addressMode != cudaAddressModeMirror || ( xfloor & 0x1 ) == 0 )
        return x - static_cast<float>( xfloor );

    // Flip coordinate for odd xfloor
    const float y = ceilf( x ) - x;
    // When the coordinate is an odd integer, ceil(x) - x returns 0, but should return near 1
    return ( y <= 0.0f ) ? firstFloatLessThanOne : y;
}

HOSTDEVICE inline unsigned int calculateWrappedTileCoord( cudaTextureAddressMode wrapMode, int coord, unsigned int levelSize, float invTileSize )
{
    return static_cast<unsigned int>( static_cast<float>( wrapPixelCoord( wrapMode, coord, levelSize ) ) * invTileSize );
}

HOSTDEVICE inline unsigned int calculateTileIndexFromTileCoords( const DemandTextureInfo& dti,
                                                                 unsigned int             mipLevel,
                                                                 unsigned int             tileX,
                                                                 unsigned int             tileY,
                                                                 unsigned int             levelWidth )
{
    const unsigned int widthInTiles     = ceilMult( levelWidth, dti.tileWidth, dti.invTileWidth );
    const unsigned int indexWithinLevel = tileY * widthInTiles + tileX;
    return indexWithinLevel + dti.numTilesBeforeLevel[mipLevel];
}

HOSTDEVICE inline unsigned int calculateTileIndex( const DemandTextureInfo& dti,
                                                   unsigned int             mipLevel,
                                                   int                      pixelX,
                                                   int                      pixelY,
                                                   unsigned int             levelWidth,
                                                   unsigned int             levelHeight )
{
    const unsigned int tileX =
        calculateWrappedTileCoord( (cudaTextureAddressMode)dti.wrapMode0, pixelX, levelWidth, dti.invTileWidth );
    const unsigned int tileY =
        calculateWrappedTileCoord( (cudaTextureAddressMode)dti.wrapMode1, pixelY, levelHeight, dti.invTileHeight );
    return calculateTileIndexFromTileCoords( dti, mipLevel, tileX, tileY, levelWidth );
}

HOSTDEVICE inline unsigned int calculateTileIndex( const DemandTextureInfo& dti, unsigned int mipLevel, float x, float y )
{
    const unsigned int levelWidth  = calculateLevelDim( mipLevel, dti.width );
    const unsigned int levelHeight = calculateLevelDim( mipLevel, dti.height );

    // We need to floor these so they don't round up when their result is between -1 and 0.
    const int pixelX = static_cast<int>( floorf( x * static_cast<float>( levelWidth ) ) );
    const int pixelY = static_cast<int>( floorf( y * static_cast<float>( levelHeight ) ) );

    return calculateTileIndex( dti, mipLevel, pixelX, pixelY, levelWidth, levelHeight );
}

/// The caller of calculateTileRequests must provide an output array with this capacity.
HOSTDEVICE constexpr inline unsigned int getCalculateTileRequestsMaxTiles()
{
    return 4;
}

HOSTDEVICE inline void calculateTileRequests( const DemandTextureInfo& dti,
                                              unsigned int             mipLevel,
                                              float                    normX,
                                              float                    normY,
                                              // The output array must have a capacity of at least MAX_TILES_CALCULATED.
                                              unsigned int* outTilesToRequest,
                                              unsigned int& outNumTilesToRequest )
{
    outNumTilesToRequest = 0;

    // If the requested miplevel is in the mip tail, return a request for tile index zero.
    if( mipLevel >= dti.mipTailFirstLevel )
    {
        outTilesToRequest[outNumTilesToRequest] = 0;
        ++outNumTilesToRequest;
        return;
    }

    const unsigned int levelWidth  = calculateLevelDim( mipLevel, dti.width );
    const unsigned int levelHeight = calculateLevelDim( mipLevel, dti.height );

    const int pixelX         = static_cast<int>( normX * levelWidth );
    const int pixelY         = static_cast<int>( normY * levelHeight );
    const int halfAnisotropy = dti.anisotropy >> 1;

    // Compute the x and y tile coordinates for left, right, top, bottom
    unsigned int xTileCoords[2];
    unsigned int yTileCoords[2];

    const cudaTextureAddressMode wrapMode0 = static_cast<cudaTextureAddressMode>( dti.wrapMode0 );
    const cudaTextureAddressMode wrapMode1 = static_cast<cudaTextureAddressMode>( dti.wrapMode0 );

    xTileCoords[0] = calculateWrappedTileCoord( wrapMode0, pixelX - halfAnisotropy, levelWidth, dti.invTileWidth );
    xTileCoords[1] = calculateWrappedTileCoord( wrapMode0, pixelX + halfAnisotropy, levelWidth, dti.invTileWidth );
    yTileCoords[0] = calculateWrappedTileCoord( wrapMode1, pixelY - halfAnisotropy, levelHeight, dti.invTileHeight );
    yTileCoords[1] = calculateWrappedTileCoord( wrapMode1, pixelY + halfAnisotropy, levelHeight, dti.invTileHeight );

    // Set the loop bounds to avoid duplicate values
    const int xmax = ( xTileCoords[0] == xTileCoords[1] ) ? 1 : 2;
    const int ymax = ( yTileCoords[0] == yTileCoords[1] ) ? 1 : 2;

    // Add each unique tileIndex to the request array
    for( int j = 0; j < ymax; ++j )
    {
        for( int i = 0; i < xmax; ++i )
        {
            outTilesToRequest[outNumTilesToRequest] =
                calculateTileIndexFromTileCoords( dti, mipLevel, xTileCoords[i], yTileCoords[j], levelWidth );
            ++outNumTilesToRequest;
        }
    }
}

// Return the mip level and pixel coordinates of the corner of the tile associated with tileIndex
HOSTDEVICE inline void unpackTileIndex( const DemandTextureInfo& dti,
                                        unsigned int             tileIndex,
                                        unsigned int&            outMipLevel,
                                        unsigned int&            outTileX,
                                        unsigned int&            outTileY )
{
    for( int mipLevel = dti.mipTailFirstLevel; mipLevel >= 0; --mipLevel )
    {
        if( ( mipLevel == 0 && dti.numPages > tileIndex ) || dti.numTilesBeforeLevel[mipLevel - 1] > tileIndex )
        {
            const unsigned int levelWidth   = calculateLevelDim( mipLevel, dti.width );
            const unsigned int widthInTiles = ceilMult( levelWidth, dti.tileWidth, dti.invTileWidth );

            const unsigned int indexInLevel = tileIndex - dti.numTilesBeforeLevel[mipLevel];
            outTileY                        = indexInLevel / widthInTiles;
            outTileX                        = indexInLevel % widthInTiles;
            outMipLevel                     = mipLevel;
            return;
        }
    }
    outMipLevel = 0;
    outTileX    = 0;
    outTileY    = 0;
}

HOSTDEVICE inline bool isMipTailIndex( unsigned int pageIndex )
{
    // Page 0 always contains the mip tail.
    return pageIndex == 0;
}

}  // namespace demandLoading
