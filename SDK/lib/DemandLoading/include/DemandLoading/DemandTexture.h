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

#include <cuda.h>
#include <vector_types.h>

#include <vector>

namespace demandLoading {

struct DemandTextureInfo;
struct TextureDescriptor;
struct TextureInfo;

/// Demand-loaded textures are created and owned by the DemandTextureManager.
class DemandTexture
{
  public:
    /// Default destructor.
    virtual ~DemandTexture() = default;

    /// Get the texture id, which is used as an index into the device-side sampler array.
    virtual unsigned int getId() const = 0;

    /// Check whether the texture has been initialized on the specified device.
    virtual bool isInitialized( unsigned int deviceIndex ) const = 0;

    /// Initialize the texture on the specified device.  When first called, this method opens the
    /// image reader that was provided to the constructor.  Returns false on error.
    virtual bool init( unsigned int deviceIndex ) = 0;

    /// Get the image info.  Valid only after the image has been initialized (e.g. opened).
    virtual const TextureInfo& getInfo() const = 0;

    /// Get device texture info.  The startPage is always valid, but the other fields are invalid
    /// the texture is initialized.
    virtual const DemandTextureInfo& getDeviceInfo() const = 0;

    /// Get the texture descriptor
    virtual const TextureDescriptor& getDescriptor() const = 0;

    /// Get the dimensions of the specified miplevel.
    virtual uint2 getMipLevelDims( unsigned int mipLevel ) const = 0;

    /// Get tile width.
    virtual unsigned int getTileWidth() const = 0;

    /// Get tile height.
    virtual unsigned int getTileHeight() const = 0;

    /// Get the first miplevel in the mip tail.
    virtual unsigned int getMipTailFirstLevel() const = 0;

    /// Get the CUDA texture object for the specified device.
    virtual CUtexObject getTextureObject( unsigned int deviceIndex ) const = 0;

    /// Read the specified tile into the given buffer, resizing it if necessary.
    virtual bool readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, std::vector<char>* buffer ) const = 0;

    /// Map the given tile backing storage and fill it with the given data.
    virtual void fillTile( unsigned int deviceIndex,
                           unsigned int mipLevel,
                           unsigned int tileX,
                           unsigned int tileY,
                           const char*  tileData,
                           size_t       tileSize ) const = 0;

    /// Read all the levels in the mip tail into the given buffer, resizing it if necessary.
    virtual bool readMipTail( std::vector<char>* buffer ) const = 0;

    /// Map the given backing storage for the mip tail and fill it with the given data.
    virtual void fillMipTail( unsigned int deviceIndex, const char* mipTailData, size_t mipTailSize ) const = 0;
};

}  // namespace demandLoading
