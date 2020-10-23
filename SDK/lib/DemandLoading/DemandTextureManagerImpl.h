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

#include "ExtensibleArray.h"
#include "PageTableManager.h"
#include "TilePool.h"
#include <DemandLoading/DemandTextureInfo.h>
#include <DemandLoading/DemandTextureManager.h>

#include <cuda.h>

#include <bitset>
#include <memory>
#include <vector>

struct OptixPagingContext;
struct PageMapping;

namespace demandLoading {

class DemandTexture;
class DemandTextureImpl;
struct DemandTextureContext;
class ImageReader;
struct TextureDescriptor;

/// DemandTextureManager demonstrates how to implement demand-loaded textures using the OptiX paging library.
class DemandTextureManagerImpl : public DemandTextureManager
{
  public:
    /// Construct demand texture manager, initializing the OptiX paging library.
    DemandTextureManagerImpl( const std::vector<unsigned int>& devices, const DemandTextureManagerConfig& config );

    /// Destroy demand texture manager, reclaiming host and device memory.
    ~DemandTextureManagerImpl() override;

    /// Create a demand-loaded texture for the given image.  The texture initially has no backing
    /// storage.  The readTile() method is invoked on the image to fill each required tile.  The
    /// ImageReader pointer is retained indefinitely.
    const DemandTexture& createTexture( std::shared_ptr<ImageReader> image, const TextureDescriptor& textureDesc ) override;

    /// Prepare for launch, updating device-side texture sampler and texture array. Returns
    /// a DemandTextureContext via result parameter.
    void launchPrepare( unsigned int deviceIndex, DemandTextureContext& demandTextureContext ) override;

    /// Process requests for missing tiles (from optixPagingMapOrRequest).
    int processRequests() override;

    // Push tile mappings to the device.  Returns the total number of new mappings.
    unsigned int pushMappings() override;

    unsigned int reservePages( unsigned int numPages ) override;

  private:
    static const unsigned int MAX_NUM_DEVICES = 32;

    struct PageRequest
    {
        unsigned int                 pageId;
        std::bitset<MAX_NUM_DEVICES> devices;
    };

    struct PerDeviceState
    {
        bool isActive = false;

        // The OptiX paging system employs a context that includes the page table, etc.
        OptixPagingContext* pagingContext = nullptr;

        // Device memory used to call OptiX paging library routines.
        // These allocations are retained to reduce allocation overhead.
        unsigned int* devRequestedPages   = nullptr;
        unsigned int* devNumPagesReturned = nullptr;
        PageMapping*  devFilledPages      = nullptr;
        unsigned int* devInvalidatedPages = nullptr;
        PageMapping*  devStalePages       = nullptr;

        std::vector<PageMapping>  filledPages;
        std::vector<PageMapping>  stalePages;
        std::vector<unsigned int> invalidatedPages;
        int                       launchNum = 0;

        ExtensibleArray<CUtexObject> textureObjects;
    };

    // Vector of demand-loaded textures, indexed by texture id.
    std::vector<DemandTextureImpl> m_textures;

    std::vector<PerDeviceState>        m_perDeviceStates;
    ExtensibleArray<DemandTextureInfo> m_textureInfo;
    PageTableManager                   m_pageTableManager;
    std::vector<char>                  m_tileBuff;
    std::vector<PageRequest>           m_pageRequests;
    std::vector<TilePool>              m_tilePools;  // one per device.
    DemandTextureManagerConfig         m_config;
    unsigned int                       m_numDevices;

    /// Get the OptiX paging library context, which is passed as a launch parameter and used to call
    /// optixPagingMapOrRequest.
    const OptixPagingContext& getPagingContext( unsigned int deviceIndex ) const
    {
        return *( m_perDeviceStates[deviceIndex].pagingContext );
    }

    // Get page requests from all active devices.
    void pullRequests();
    // Get page requests from the device (via optixPagingPullRequests).
    std::vector<unsigned int> pullRequests( PerDeviceState& state );

    void processStartPageRequest( const PageRequest& request, DemandTexture* texture );
    void processTileRequest( const PageRequest& request, DemandTexture* texture );
    void initTexture( unsigned int deviceIndex, DemandTexture* texture );
};

}  // namespace demandLoading
