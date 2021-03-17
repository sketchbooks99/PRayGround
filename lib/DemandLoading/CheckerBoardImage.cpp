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
#include <DemandLoading/CheckerBoardImage.h>

#include "Exception.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>  // for make_float4

namespace demandLoading {

CheckerBoardImage::CheckerBoardImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps )
    : m_squaresPerSide( squaresPerSide )
    , m_info{width, height, CU_AD_FORMAT_FLOAT, /*numChannels=*/4, /*numMipLevels=*/0}
{
    const unsigned int dim = std::max( width, height );
    m_info.numMipLevels = useMipmaps ? 1 + static_cast<unsigned int>( std::ceil( std::log2f( static_cast<float>( dim ) ) ) ) : 1;

    // Use a different color per miplevel.
    std::vector<float4> colors{
        {255, 0, 0, 0},    // red
        {255, 127, 0, 0},  // orange
        {255, 255, 0, 0},  // yellow
        {0, 255, 0, 0},    // green
        {0, 0, 255, 0},    // blue
        {127, 0, 0, 0},    // dark red
        {127, 63, 0, 0},   // dark orange
        {127, 127, 0, 0},  // dark yellow
        {0, 127, 0, 0},    // dark green
        {0, 0, 127, 0},    // dark blue
    };
    // Normalize the miplevel colors to [0,1]
    for( float4& color : colors )
    {
        color.x /= 255.f;
        color.y /= 255.f;
        color.z /= 255.f;
    }
    m_mipLevelColors.swap( colors );
}

bool CheckerBoardImage::open( TextureInfo* info )
{
    if( info != nullptr )
        *info = m_info;
    return true;
}


// Determine if checkerboard square is "odd" or "even" (e.g. red/black).
bool isOddChecker( int x, int y )
{
    return ( ( x + y ) & 1 ) != 0;
}


bool CheckerBoardImage::readTile( char* dest, unsigned int mipLevel, unsigned int tileX, unsigned int tileY, unsigned int tileWidth, unsigned int tileHeight )
{
    if( mipLevel >= m_info.numMipLevels )
        return false;

    const float4       black    = make_float4( 0.f, 0.f, 0.f, 0.f );
    const float4       color    = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];
    const unsigned int gridSize = std::max( 1U, ( m_info.width / m_squaresPerSide ) >> mipLevel );

    const unsigned int startX = tileX * tileWidth;
    const unsigned int startY = tileY * tileHeight;

    const unsigned int rowPitch = tileWidth * m_info.numChannels * getBytesPerChannel( m_info.format );

    for( unsigned int destY = 0; destY < tileHeight; ++destY )
    {
        float4* row = reinterpret_cast<float4*>( dest + destY * rowPitch );
        for( unsigned int destX = 0; destX < tileWidth; ++destX )
        {
            const int srcX = destX + startX;
            const int srcY = destY + startY;

            const bool odd = isOddChecker( srcX / gridSize, srcY / gridSize );
            row[destX]     = odd ? black : color;
        }
    }
    return true;
}

bool CheckerBoardImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height )
{
    // Create a checkerboard pattern with a color based on the miplevel.
    const float4       black    = make_float4( 0.f, 0.f, 0.f, 0.f );
    const float4       color    = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];
    const unsigned int gridSize = std::max( 1U, ( m_info.width / m_squaresPerSide ) >> mipLevel );

    float4* pixels = reinterpret_cast<float4*>( dest );

    for( unsigned int y = 0; y < height; ++y )
    {
        float4* row = pixels + y * width;
        for( unsigned int x = 0; x < width; ++x )
        {
            const bool odd = isOddChecker( x / gridSize, y / gridSize );
            row[x]         = odd ? black : color;
        }
    }

    return true;
}

}  // namespace demandLoading
