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

#include "TilePool.h"

#include "Exception.h"
#include "Math.h"

namespace demandLoading {

TilePool::TilePool( unsigned int deviceIndex )
    : m_deviceIndex( deviceIndex )
{
    // Use the recommended allocation granularity as the arena size.  Typically this gives 32 tiles per arena.
    CUmemAllocationProp prop{};
    prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( m_deviceIndex )};
    prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
    DEMAND_CUDA_CHECK( cuMemGetAllocationGranularity( &m_arenaSize, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ) );

    DEMAND_ASSERT( m_arenaSize >= TILE_SIZE );
}

TilePool::~TilePool()
{
    for( CUmemGenericAllocationHandle arena : m_arenas )
    {
        DEMAND_CUDA_CHECK( cuMemRelease( arena ) );
    }
}

void TilePool::allocate( size_t numBytes, CUmemGenericAllocationHandle* handle, size_t* offset )
{
    // Set current device.
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Allocate an integral number of tiles.
    numBytes = idivCeil( numBytes, TILE_SIZE ) * TILE_SIZE;

    // Create a new arena if necessary.
    if( m_arenas.empty() || m_offset + numBytes > m_arenaSize )
    {
        m_arenas.push_back( createArena() );
        m_offset = 0;
    }

    // Return the current arena handle and offset.
    *handle = m_arenas.back();
    *offset = m_offset;

    // Increment the current offset.
    m_offset += numBytes;
}

CUmemGenericAllocationHandle TilePool::createArena() const
{
    CUmemAllocationProp prop{};
    prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( m_deviceIndex )};
    prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
    CUmemGenericAllocationHandle arena;
    DEMAND_CUDA_CHECK( cuMemCreate( &arena, m_arenaSize, &prop, 0 ) );
    return arena;
}

}  // namespace demandLoading
