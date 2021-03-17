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

#include "DemandTextureImpl.h"
#include "Math.h"
#include "PageTableManager.h"
#include "TilePool.h"
#include <DemandLoading/ImageReader.h>

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace demandLoading {

DemandTextureImpl::DemandTextureImpl( unsigned int                 id,
                                      unsigned int                 maxNumDevices,
                                      const TextureDescriptor&     descriptor,
                                      std::shared_ptr<ImageReader> image,
                                      std::vector<TilePool>*       tilePools,
                                      PageTableManager*            pageTableManager )

    : m_id( id )
    , m_descriptor( descriptor )
    , m_image( image )
    , m_tilePools( tilePools )
    , m_pageTableManager( pageTableManager )
{
    // Initially we reserve a single page, since we don't know how many tiles are required until the
    // texture is fully initialized.  The other fields of the device texture info are not valid
    // until the texture is initialized.
    m_deviceInfo.startPage = m_pageTableManager->reserve( 1, m_id );
    m_deviceInfo.numPages  = 1;

    // Construct per-device sparse textures.
    m_textures.reserve( maxNumDevices );
    for( unsigned int i = 0; i < maxNumDevices; ++i )
        m_textures.emplace_back( i );
}

bool DemandTextureImpl::init( unsigned int deviceIndex )
{
    // Open the image if necessary, fetching the dimensions and other info.
    if( !m_isInitialized )
    {
        if( !m_image->open( &m_info ) )
            return false;
    }

    // Initialize the sparse texture for the specified device.
    DEMAND_ASSERT( deviceIndex < m_textures.size() );
    SparseTexture& sparseTexture = m_textures[deviceIndex];
    sparseTexture.init( m_descriptor, m_info );

    if( !m_isInitialized )
    {
        m_isInitialized = true;

        // Retain various properties for subsequent use.  (They're the same on all devices.)
        m_tileWidth         = sparseTexture.getTileWidth();
        m_tileHeight        = sparseTexture.getTileHeight();
        m_mipTailFirstLevel = sparseTexture.getMipTailFirstLevel();
        m_mipTailSize       = m_mipTailFirstLevel < m_info.numMipLevels ? sparseTexture.getMipTailSize() : 0;

        // Verify that the tile size agrees with TilePool.
        DEMAND_ASSERT( m_tileWidth * m_tileHeight * getBytesPerChannel( getInfo().format ) <= TilePool::TILE_SIZE );

        // Record the dimensions of each miplevel.
        const unsigned int numMipLevels = getInfo().numMipLevels;
        m_mipLevelDims.resize( numMipLevels );
        for( unsigned int i = 0; i < numMipLevels; ++i )
        {
            m_mipLevelDims[i] = sparseTexture.getMipLevelDims( i );
        }

        // Update device texture info.
        initDemandTextureInfo();
    }
    return true;
}

void DemandTextureImpl::initDemandTextureInfo()
{
    // Dimensions
    m_deviceInfo.width  = getInfo().width;
    m_deviceInfo.height = getInfo().height;

    // Filtering and status.
    m_deviceInfo.mipLevels         = getInfo().numMipLevels;
    m_deviceInfo.mipTailFirstLevel = getMipTailFirstLevel();
    m_deviceInfo.wrapMode0         = static_cast<int>( getDescriptor().addressMode[0] );
    m_deviceInfo.wrapMode1         = static_cast<int>( getDescriptor().addressMode[1] );
    m_deviceInfo.anisotropy        = getDescriptor().maxAnisotropy;
    m_deviceInfo.mipmapFilterMode  = static_cast<int>( getDescriptor().mipmapFilterMode );
    m_deviceInfo.isInitialized     = 1;

    // Tiling
    m_deviceInfo.tileWidth  = getTileWidth();
    m_deviceInfo.tileHeight = getTileHeight();

    // Calculate number of tiles.
    m_deviceInfo.numPages  = 1;  // for the mip tail.
    unsigned int lastLevel = std::min( getMipTailFirstLevel(), getInfo().numMipLevels );
    for( unsigned int i = 0; i < lastLevel; ++i )
    {
        m_deviceInfo.numPages += getNumTilesInLevel( i );
    }

    // Reserve a range of page table entries, one per tile.
    m_deviceInfo.startPage = m_pageTableManager->reserve( m_deviceInfo.numPages, getId() );

    // Precomputed reciprocals
    m_deviceInfo.invWidth      = 1.0f / static_cast<float>( m_deviceInfo.width );
    m_deviceInfo.invHeight     = 1.0f / static_cast<float>( m_deviceInfo.height );
    m_deviceInfo.invTileWidth  = 1.0f / static_cast<float>( m_deviceInfo.tileWidth );
    m_deviceInfo.invTileHeight = 1.0f / static_cast<float>( m_deviceInfo.tileHeight );
    m_deviceInfo.invAnisotropy = 1.0f / static_cast<float>( m_deviceInfo.anisotropy );

    // init numTilesBeforeLevel (assuming array is 0 initialized)
    memset( m_deviceInfo.numTilesBeforeLevel, 0, MAX_TILE_LEVELS * sizeof( unsigned int ) );

    for( int level = static_cast<int>( lastLevel ) - 1; level >= 0; --level )
    {
        m_deviceInfo.numTilesBeforeLevel[level] = m_deviceInfo.numTilesBeforeLevel[level + 1] + getNumTilesInLevel( level + 1 );
    }
}

unsigned int DemandTextureImpl::getNumTilesInLevel( unsigned int mipLevel ) const
{
    if( mipLevel == getMipTailFirstLevel() )
        return 1;
    if( mipLevel > getMipTailFirstLevel() || mipLevel >= getInfo().numMipLevels )
        return 0;
    DEMAND_ASSERT( mipLevel < m_mipLevelDims.size() );
    const unsigned int tilesWide = idivCeil( m_mipLevelDims[mipLevel].x, m_tileWidth );
    const unsigned int tilesHigh = idivCeil( m_mipLevelDims[mipLevel].y, m_tileHeight );
    return tilesWide * tilesHigh;
}


bool DemandTextureImpl::readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, std::vector<char>* buffer ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );

    // Resize buffer if necessary.
    const unsigned int bytesPerPixel = getBytesPerChannel( getInfo().format ) * getInfo().numChannels;
    const unsigned int bytesPerTile  = getTileWidth() * getTileHeight() * bytesPerPixel;
    buffer->resize( bytesPerTile );

    return m_image->readTile( buffer->data(), mipLevel, tileX, tileY, getTileWidth(), getTileHeight() );
}


void DemandTextureImpl::fillTile( unsigned int deviceIndex,
                                  unsigned int mipLevel,
                                  unsigned int tileX,
                                  unsigned int tileY,
                                  const char*  tileData,
                                  size_t       tileSize ) const
{
    DEMAND_ASSERT( deviceIndex < m_tilePools->size() );
    DEMAND_ASSERT( deviceIndex < m_textures.size() );
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );

    // Allocate tile on device.
    DEMAND_ASSERT( tileSize <= TilePool::TILE_SIZE );
    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    ( *m_tilePools )[deviceIndex].allocate( tileSize, &handle, &offset );

    m_textures[deviceIndex].fillTile( mipLevel, tileX, tileY, tileData, tileSize, handle, offset );
}


bool DemandTextureImpl::readMipTail( std::vector<char>* buffer ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( getMipTailFirstLevel() < m_info.numMipLevels );

    // Resize buffer if necessary.
    buffer->resize( m_mipTailSize );

    const unsigned int pixelSize = getInfo().numChannels * getBytesPerChannel( getInfo().format );
    return m_image->readMipTail( buffer->data(), getMipTailFirstLevel(), getInfo().numMipLevels, m_mipLevelDims.data(), pixelSize );
}


void DemandTextureImpl::fillMipTail( unsigned int deviceIndex, const char* mipTailData, size_t mipTailSize ) const
{
    DEMAND_ASSERT( deviceIndex < m_tilePools->size() );
    DEMAND_ASSERT( deviceIndex < m_textures.size() );
    DEMAND_ASSERT( getMipTailFirstLevel() < m_info.numMipLevels );

    // Allocate backing storage on device, which might require multiple tiles.
    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    ( *m_tilePools )[deviceIndex].allocate( mipTailSize, &handle, &offset );

    m_textures[deviceIndex].fillMipTail( mipTailData, mipTailSize, handle, offset );
}

}  // namespace demandLoading
