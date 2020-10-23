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

#include "DemandTextureManagerImpl.h"

#include "DemandTextureImpl.h"
#include "Exception.h"
#include <DemandLoading/DemandTextureContext.h>
#include <DemandLoading/ImageReader.h>
#include <DemandLoading/Tex2D.h>

#include <optixPaging/optixPaging.h>

#include <algorithm>

namespace demandLoading {

// Construct demand texture manager, initializing the OptiX paging library.
DemandTextureManagerImpl::DemandTextureManagerImpl( const std::vector<unsigned int>& devices, const DemandTextureManagerConfig& config )
    : m_perDeviceStates( devices.size() )
    , m_textureInfo( static_cast<unsigned int>( devices.size() ) )
    , m_pageTableManager( config.numPages )
    , m_config( config )
    , m_numDevices( static_cast<unsigned int>( devices.size() ) )
{
    unsigned int numCapableDevices = 0;
    for( unsigned int currDevice : devices )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( currDevice ) );

        // Skip device if it doesn't support sparse textures.
        CUdevice device;
        DEMAND_CUDA_CHECK( cuDeviceGet( &device, currDevice ) );
        int sparseSupport = 0;

        DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &sparseSupport, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, device ) );
        if( !sparseSupport )
            continue;
        ++numCapableDevices;

        PerDeviceState& currState = m_perDeviceStates[currDevice];
        currState.isActive        = true;

        // Configure the paging library.
        OptixPagingOptions options{m_config.numPages, m_config.numPages};
        optixPagingCreate( &options, &currState.pagingContext );
        OptixPagingSizes sizes{};
        optixPagingCalculateSizes( options.initialVaSizeInPages, sizes );

        // Allocate device memory required by the paging library.
        DEMAND_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &currState.pagingContext->pageTable ), sizes.pageTableSizeInBytes ) );
        DEMAND_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &currState.pagingContext->usageBits ), sizes.usageBitsSizeInBytes ) );
        optixPagingSetup( currState.pagingContext, sizes, 1 );

        // Allocate device memory that is used to call paging library routines.
        // These allocations are retained to reduce allocation overhead.
        DEMAND_CUDA_CHECK( cudaMalloc( &currState.devRequestedPages, m_config.maxRequestedPages * sizeof( unsigned int ) ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &currState.devNumPagesReturned, 3 * sizeof( unsigned int ) ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &currState.devFilledPages, m_config.maxFilledPages * sizeof( PageMapping ) ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &currState.devInvalidatedPages, m_config.maxInvalidatedPages * sizeof( unsigned int ) ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &currState.devStalePages, m_config.maxStalePages * sizeof( PageMapping ) ) );

        // The array of texture objects is per-device.  It's represented by an ExtensibleArray of
        // devices in order to leverage the dirty/grown logic in that class when synchronizing.
        // It's not a great fit, however, because ExtensibleArray usually represents a single host
        // array that is the same on all devices, whereas in this case we have a separate
        // ExtensibleArray per device.
        currState.textureObjects = ExtensibleArray<CUtexObject>( static_cast<unsigned int>( devices.size() ) );
    }
    if( numCapableDevices == 0 )
        throw Exception( "No devices that support CUDA sparse textures were found (sm_60+ required)." );

    // Initialize tile pools.  Nothing is allocated yet.
    m_tilePools.reserve( devices.size() );
    for( unsigned int deviceIndex : devices )
    {
        m_tilePools.emplace_back( deviceIndex );
    }
}

DemandTextureManagerImpl::~DemandTextureManagerImpl()
{
    for( PerDeviceState& state : m_perDeviceStates )
    {
        if( state.isActive )
        {
            // Free device memory and destroy the paging system.
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.pagingContext->pageTable ) );
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.pagingContext->usageBits ) );
            optixPagingDestroy( state.pagingContext );

            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.devRequestedPages ) );
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.devNumPagesReturned ) );
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.devFilledPages ) );
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.devInvalidatedPages ) );
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( state.devStalePages ) );
        }
    }
}

const DemandTexture& DemandTextureManagerImpl::createTexture( std::shared_ptr<ImageReader> imageReader,
                                                              const TextureDescriptor&     textureDesc )
{
    // The texture id will be the next index in the texture array.
    unsigned int textureId = static_cast<unsigned int>( m_textures.size() );

    // Add new texture to the end of the list of textures.  The texture holds a pointer to the
    // image, from which tile data is obtained on demand.
    m_textures.emplace_back( textureId, m_numDevices, textureDesc, imageReader, &m_tilePools, &m_pageTableManager );
    const DemandTexture& texture = m_textures.back();

    // Add entry to DemandTextureInfo array, which is indexed by texture id.  Most of the fields are
    // not valid until the texture is fully initialized.
    m_textureInfo.pushBack( texture.getDeviceInfo() );

    return texture;
}

// Prepare for launch, updating device-side texture sampler and texture array. Returns
// a DemandTextureContext via result parameter.
void DemandTextureManagerImpl::launchPrepare( unsigned int deviceIndex, DemandTextureContext& demandTextureContext )
{
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    demandTextureContext.m_pagingContext = getPagingContext( deviceIndex );
    demandTextureContext.m_textureInfos  = m_textureInfo.synchronize( deviceIndex );

    PerDeviceState& state           = m_perDeviceStates[deviceIndex];
    demandTextureContext.m_textures = state.textureObjects.synchronize( deviceIndex );
}

// Get page requests from all active devices.  The requests are recorded in a member variable to amortize allocation overhead.
void DemandTextureManagerImpl::pullRequests()
{
    struct DevicePageRequest
    {
        unsigned int pageId;
        unsigned int deviceIndex;
    };
    std::vector<DevicePageRequest> allRequests;

    for( unsigned int deviceIndex = 0; deviceIndex < static_cast<unsigned int>( m_perDeviceStates.size() ); ++deviceIndex )
    {
        PerDeviceState& currState = m_perDeviceStates[deviceIndex];
        if( currState.isActive )
        {
            // Get the tile requests for this device.
            DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
            std::vector<unsigned int> pageRequests( pullRequests( currState ) );

            // Bundle the page requests with the device index.
            allRequests.reserve( allRequests.size() + pageRequests.size() );
            for( unsigned int page : pageRequests )
            {
                allRequests.push_back( DevicePageRequest{page, deviceIndex} );
            }
        }
    }

    // Sort the requests by page number.  This ensures that all the requests for a particular
    // tile are adjacent in the request list.
    std::sort( allRequests.begin(), allRequests.end(),
               []( const DevicePageRequest& a, const DevicePageRequest& b ) { return a.pageId < b.pageId; } );

    // Bundle requests for the same page on multiple devices into a single request that encodes the
    // devices in a bitmask.  The request vector is stored in a member variable to amortize
    // allocation overhead.
    m_pageRequests.clear();
    const size_t numRequests = allRequests.size();
    for( size_t i = 0; i < numRequests; /* nop */ )
    {
        // Push a new page request.
        const DevicePageRequest& request = allRequests[i];
        m_pageRequests.push_back( {request.pageId, std::bitset<32>()} );

        // Record each device that requested this page.
        for( ; i < numRequests && allRequests[i].pageId == request.pageId; ++i )
        {
            m_pageRequests.back().devices.set( allRequests[i].deviceIndex );
        }
    }
}

// Get page requests from the specified device (via optixPagingPullRequests).
std::vector<unsigned int> DemandTextureManagerImpl::pullRequests( PerDeviceState& state )
{
    // Get a list of requested page ids, along with lists of stale and evictable pages (which are currently unused).
    unsigned int* evictablePages    = nullptr;
    unsigned int  numEvictablePages = 0;
    optixPagingPullRequests( state.pagingContext, state.devRequestedPages, m_config.maxRequestedPages, state.devStalePages,
                             m_config.maxStalePages, evictablePages, numEvictablePages, state.devNumPagesReturned );

    // Get the sizes of the requested, stale, and evictable page lists.
    unsigned int numReturned[3] = {0};
    DEMAND_CUDA_CHECK( cudaMemcpy( &numReturned[0], state.devNumPagesReturned, 3 * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );

    // Copy the requested page list from this device.
    unsigned int              numRequests = numReturned[0];
    std::vector<unsigned int> requestedPages( numRequests );
    if( numRequests > 0 )
    {
        DEMAND_CUDA_CHECK( cudaMemcpy( requestedPages.data(), state.devRequestedPages,
                                       numRequests * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
    }
    return requestedPages;
}

// Process page requests.
int DemandTextureManagerImpl::processRequests()
{
    pullRequests();

    for( const PageRequest& request : m_pageRequests )
    {
        // Find the texture id for this request via the PageTableManager, which keeps track of the
        // range of page table entries reserved for each texture.
        const unsigned int textureId = m_pageTableManager.getResource( request.pageId );

        // The texture id serves as an index.
        DemandTexture*           texture = &m_textures[textureId];
        const DemandTextureInfo& info    = m_textureInfo.get( textureId );

        // Process the request.
        if( request.pageId == info.startPage )
            processStartPageRequest( request, texture );
        else
            processTileRequest( request, texture );
    }

    // Push the new page mappings to the device.
    return pushMappings();
}

// The start page is requested (1) if the texture is uninitialized, or (2) if a miplevel in the mip tail is required.
void DemandTextureManagerImpl::processStartPageRequest( const PageRequest& request, DemandTexture* texture )
{
    bool mipTailLoaded = false;
    for( unsigned int deviceIndex = 0; deviceIndex < MAX_NUM_DEVICES; ++deviceIndex )
    {
        if( !request.devices[deviceIndex] )
            continue;

        if( texture->isInitialized( deviceIndex ) )
        {
            if( !mipTailLoaded )
            {
                // Read the mip tail (into a member variable, which amortizes allocation overhead).
                const bool ok = texture->readMipTail( &m_tileBuff );
                DEMAND_ASSERT_MSG( ok, "readMipTail call failed" );
                mipTailLoaded = true;
            }
            texture->fillMipTail( deviceIndex, m_tileBuff.data(), m_tileBuff.size() );
        }
        else
        {
            initTexture( deviceIndex, texture );
        }

        // Push page mapping for the requested page.
        PerDeviceState& state = m_perDeviceStates[deviceIndex];
        state.filledPages.push_back( PageMapping{request.pageId, 1 /*arbitrary*/} );
    }
}

// Initialize texture in preparation for reading tile data.
void DemandTextureManagerImpl::initTexture( unsigned int deviceIndex, DemandTexture* texture )
{
    // Initialize the texture, reading image info from file header.
    const bool ok = texture->init( deviceIndex );
    DEMAND_ASSERT_MSG( ok, "ImageReader::init() failed" );

    // Update device texture info.
    const unsigned int textureId = texture->getId();
    m_textureInfo.set( textureId, texture->getDeviceInfo() );

    // Record the texture object, which will be synchronized to the device.
    PerDeviceState& state = m_perDeviceStates[deviceIndex];
    state.textureObjects.expand( textureId );
    state.textureObjects.set( textureId, texture->getTextureObject( deviceIndex ) );
}

void DemandTextureManagerImpl::processTileRequest( const PageRequest& request, DemandTexture* texture )
{
    // Unpack tile index into miplevel and tile coordinates.
    const unsigned int tileIndex = request.pageId - texture->getDeviceInfo().startPage;
    unsigned int       mipLevel;
    unsigned int       tileX;
    unsigned int       tileY;
    unpackTileIndex( texture->getDeviceInfo(), tileIndex, mipLevel, tileX, tileY );

    // Read the tile from disk into the tile buffer (which is a member variable, in order to amortize allocation overhead).
    const bool ok = texture->readTile( mipLevel, tileX, tileY, &m_tileBuff );
    DEMAND_ASSERT_MSG( ok, "readTile call failed" );

    // Fill all requests for the same page from different devices.
    for( unsigned int deviceIndex = 0; deviceIndex < MAX_NUM_DEVICES; ++deviceIndex )
    {
        if( !request.devices[deviceIndex] )
            continue;

        texture->fillTile( deviceIndex, mipLevel, tileX, tileY, m_tileBuff.data(), m_tileBuff.size() );

        // Record the new page mapping.  Note that we don't currently use the value in the page table
        // entry.  Mapping to a boolean would suffice.
        PerDeviceState& state = m_perDeviceStates[deviceIndex];
        state.filledPages.push_back( PageMapping{request.pageId, 1 /*arbitrary*/} );
    }
}

// Push tile mappings to the device.  Returns the total number of new mappings.
unsigned int DemandTextureManagerImpl::pushMappings()
{
    unsigned int totalRequestsFilled = 0;
    for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); ++deviceIndex )
    {
        PerDeviceState& currState = m_perDeviceStates[deviceIndex];
        if( currState.isActive )
        {
            const unsigned int numFilledPages =
                std::min( static_cast<unsigned int>( currState.filledPages.size() ), m_config.maxFilledPages );
            const unsigned int numInvalidatedPages =
                std::min( static_cast<unsigned int>( currState.invalidatedPages.size() ), m_config.maxInvalidatedPages );
            totalRequestsFilled += numFilledPages;
            DEMAND_CUDA_CHECK( cudaSetDevice( static_cast<unsigned int>( deviceIndex ) ) );
            DEMAND_CUDA_CHECK( cudaMemcpy( currState.devInvalidatedPages, currState.invalidatedPages.data(),
                                           numInvalidatedPages * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );
            DEMAND_CUDA_CHECK( cudaMemcpy( currState.devFilledPages, currState.filledPages.data(),
                                           numFilledPages * sizeof( PageMapping ), cudaMemcpyHostToDevice ) );
            optixPagingPushMappings( currState.pagingContext, currState.devFilledPages, numFilledPages,
                                     currState.devInvalidatedPages, numInvalidatedPages );
            currState.filledPages.clear();
            currState.invalidatedPages.clear();
        }
    }
    return totalRequestsFilled;
}

unsigned DemandTextureManagerImpl::reservePages( unsigned numPages )
{
    const unsigned int noResourceId = 0xffffffff;
    return m_pageTableManager.reserve( numPages, noResourceId );
}

DemandTextureManager* createDemandTextureManager( const std::vector<unsigned int>& devices, const DemandTextureManagerConfig& config )
{
    return new DemandTextureManagerImpl( devices, config );
}

void destroyDemandTextureManager( DemandTextureManager* manager )
{
    delete manager;
}

}  // namespace demandLoading
