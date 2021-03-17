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

#include <DemandLoading/DemandTextureInfo.h>
#include <DemandLoading/Exception.h>
#include <DemandLoading/ImageReader.h>
#include <DemandLoading/SparseTexture.h>
#include <DemandLoading/TextureDescriptor.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace demandLoading {

class ImageReader;
class PageTableManager;
class TilePool;

/// Demand-loaded textures are created by the DemandTextureManager.
class DemandTexture
{
  public:
    /// Default constructor.
    DemandTexture() {}

    /// Construct demand loaded texture with the specified id (which is used as an index into the
    /// device-side sampler array) the the given descriptor (which specifies the wrap mode, filter
    /// mode, etc.).  The given image reader is retained and used by subsequent readTile() calls.
    DemandTexture( unsigned int                 id,
                   unsigned int                 maxNumDevices,
                   const TextureDescriptor&     descriptor,
                   std::unique_ptr<ImageReader> image,
                   std::vector<TilePool>*       tilePools,
                   PageTableManager*            pageTableManager );

    /// Get the texture id, which is used as an index into the device-side sampler array.
    unsigned int getId() const { return m_id; }

    /// Check whether the texture has been initialized on the specified device.
    bool isInitialized( unsigned int deviceIndex ) const
    {
        DEMAND_ASSERT( deviceIndex < m_textures.size() );
        return m_isInitialized && m_textures[deviceIndex].isInitialized();
    }

    /// Initialize the texture on the specified device.  When first called, this method opens the
    /// image reader that was provided to the constructor.  Returns false on error.
    bool init( unsigned int deviceIndex );

    /// Get the image info.  Valid only after the image has been initialized (e.g. opened).
    const TextureInfo& getInfo() const
    {
        DEMAND_ASSERT( m_isInitialized );
        return m_info;
    }

    /// Get device texture info.  The startPage is always valid, but the other fields are invalid
    /// the texture is initialized.
    const DemandTextureInfo& getDeviceInfo() const { return m_deviceInfo; }

    /// Get the texture descriptor
    const TextureDescriptor& getDescriptor() const { return m_descriptor; }

    /// Get the dimensions of the specified miplevel.
    uint2 getMipLevelDims( unsigned int mipLevel ) const
    {
        DEMAND_ASSERT( m_isInitialized );
        DEMAND_ASSERT( mipLevel < m_mipLevelDims.size() );
        return m_mipLevelDims[mipLevel];
    }

    /// Get tile width.
    unsigned int getTileWidth() const
    {
        DEMAND_ASSERT( m_isInitialized );
        return m_tileWidth;
    }

    /// Get tile height.
    unsigned int getTileHeight() const
    {
        DEMAND_ASSERT( m_isInitialized );
        return m_tileHeight;
    }

    /// Get the first miplevel in the mip tail.
    unsigned int getMipTailFirstLevel() const
    {
        DEMAND_ASSERT( m_isInitialized );
        return m_mipTailFirstLevel;
    }

    /// Get the CUDA texture object for the specified device.
    CUtexObject getTextureObject( unsigned int deviceIndex ) const
    {
        DEMAND_ASSERT( m_isInitialized );
        DEMAND_ASSERT( deviceIndex < m_textures.size() );
        return m_textures[deviceIndex].getTextureObject();
    }

    /// Read the specified tile into the given buffer, resizing it if necessary.
    bool readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, std::vector<char>* buffer ) const;

    /// Map the given tile backing storage and fill it with the given data.
    void fillTile( unsigned int deviceIndex, unsigned int mipLevel, unsigned int tileX, unsigned int tileY, const char* tileData, size_t tileSize ) const;

    /// Read all the levels in the mip tail into the given buffer, resizing it if necessary.
    bool readMipTail( std::vector<char>* buffer ) const;

    /// Map the given backing storage for the mip tail and fill it with the given data.
    void fillMipTail( unsigned int deviceIndex, const char* mipTailData, size_t mipTailSize ) const;

  private:
    // The texture identifier is used as an index into the device-side sampler array.
    unsigned int m_id = 0;

    // The texture descriptor specifies wrap and filtering modes, etc.
    TextureDescriptor m_descriptor{};

    // The image provides a read() method that fills requested miplevels.
    std::unique_ptr<ImageReader> m_image;

    // Per-device tile pools provides backing storage.  (Owned by DemandTextureManager.)
    std::vector<TilePool>* m_tilePools;

    // The page table manager provides a range of page ids, one per tile (plus one for the mip tail).
    PageTableManager* m_pageTableManager;

    // The image is lazily opened.
    bool m_isInitialized = false;

    // Image info, including dimensions and format.  Not valid until the image is initialized.
    TextureInfo       m_info{};
    DemandTextureInfo m_deviceInfo{};
    unsigned int      m_tileWidth         = 0;
    unsigned int      m_tileHeight        = 0;
    unsigned int      m_mipTailFirstLevel = 0;
    size_t            m_mipTailSize       = 0;

    // Sparse textures (one per device).
    std::vector<SparseTexture> m_textures;

    // Miplevel dimensions.
    std::vector<uint2> m_mipLevelDims;

    void         initDemandTextureInfo();
    unsigned int getNumTilesInLevel( unsigned int mipLevel ) const;
};

}  // namespace demandLoading
