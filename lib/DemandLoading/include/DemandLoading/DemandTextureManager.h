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

#include <memory>
#include <vector>

namespace demandLoading {

class DemandTexture;
struct DemandTextureContext;
class DemandTextureManagerImpl;
class ImageReader;
struct TextureDescriptor;

struct DemandTextureManagerConfig
{
    unsigned int numPages;             // max virtual pages
    unsigned int maxRequestedPages;    // max requests to pull from device
    unsigned int maxFilledPages;       // num slots to push mappings back to device
    unsigned int maxStalePages;        // max stale pages to pull from device
    unsigned int maxInvalidatedPages;  // max slots to push invalidated pages back to device
};

/// DemandTextureManager demonstrates how to implement demand-loaded textures using the OptiX paging library.
class DemandTextureManager
{
  public:
    /// Base class destructor.
    virtual ~DemandTextureManager() = default;

    /// Create a demand-loaded texture for the given image.  The texture initially has no backing
    /// storage.  The readTile() method is invoked on the image to fill each required tile.  The
    /// ImageReader pointer is retained indefinitely.
    virtual const DemandTexture& createTexture( std::shared_ptr<ImageReader> image, const TextureDescriptor& textureDesc ) = 0;

    /// Prepare for launch, updating device-side texture sampler and texture array. Returns
    /// a DemandTextureContext via result parameter.
    virtual void launchPrepare( unsigned int deviceIndex, DemandTextureContext& demandTextureContext ) = 0;

    /// Process requests for missing tiles (from optixPagingMapOrRequest).
    virtual int processRequests() = 0;

    // Push tile mappings to the device.  Returns the total number of new mappings.
    virtual unsigned int pushMappings() = 0;

    virtual unsigned int reservePages( unsigned int numPages ) = 0;
};

/// Factory function to create a demand texture manager for the given configuration on the given devices.
DemandTextureManager* createDemandTextureManager( const std::vector<unsigned int>& devices, const DemandTextureManagerConfig& config );

/// Function to destroy a demand texture manager.
void destroyDemandTextureManager( DemandTextureManager* manager );

}  // namespace demandLoading
