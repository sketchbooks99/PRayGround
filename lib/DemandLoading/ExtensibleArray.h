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

#include "Exception.h"

#include <cuda_runtime.h>

#include <vector>

namespace demandLoading {

/// ExtensibleArray<T> encapsulates the synchronization of a variable-length
/// array (e.g. a sampler array).
template <typename T>
class ExtensibleArray
{
  public:
    /// Construct ExtensibleArray for the specified number of devices.
    explicit ExtensibleArray( unsigned int numDevices )
        : m_devArrays( numDevices, nullptr )
        , m_devArrayLengths( numDevices, 0 )
    {
    }

    /// Construct ExtensibleArray for a single device
    ExtensibleArray()
        : m_devArrays( 1 /*numDevices*/, nullptr )
        , m_devArrayLengths( 1 /*numDevices*/, 0 )
    {
    }

    /// The destructor is virtual.
    virtual ~ExtensibleArray()
    {
        for( T* devArray : m_devArrays )
        {
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( devArray ) );
        }
    }

    /// Get the specified element from the host array.
    const T& get( unsigned int index )
    {
        DEMAND_ASSERT( index < m_hostArray.size() );
        return m_hostArray[index];
    }

    /// Set the specified element in the host array.  The array is marked "dirty", ensuring that
    /// synchronize() will re-copy the array to device memory.
    void set( unsigned int index, const T& src )
    {
        DEMAND_ASSERT( index < m_hostArray.size() );
        m_hostArray[index] = src;
        m_isDirty          = true;
    }

    /// Add an element to the end of the host array.
    void pushBack( const T& element ) { m_hostArray.push_back( element ); }

    /// Expand to accomodate an entry at the given index.
    void expand( unsigned int index )
    {
        if( index >= m_hostArray.size() )
        {
            m_hostArray.resize( index + 1 );
        }
    }

    /// Synchronize host array to the specified device if necessary.  Returns the array's device pointer.
    T* synchronize( unsigned int deviceIndex )
    {
        DEMAND_ASSERT( deviceIndex < m_devArrayLengths.size() );
        const unsigned int devArrayLength = m_devArrayLengths[deviceIndex];
        bool               hasGrown       = m_hostArray.size() > devArrayLength;
        if( m_isDirty && !hasGrown )
        {
            // Copy host array to device.
            T*& devArray = m_devArrays[deviceIndex];
            DEMAND_CUDA_CHECK( cudaMemcpy( devArray, m_hostArray.data(), m_hostArray.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
            return devArray;
        }
        else if( hasGrown )
        {
            // Reallocate device array.
            T*  oldArray = m_devArrays[deviceIndex];
            T*& newArray = m_devArrays[deviceIndex];
            DEMAND_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &newArray ), m_hostArray.size() * sizeof( T ) ) );

            if( m_isDirty )
            {
                // Copy host array to device.
                DEMAND_CUDA_CHECK( cudaMemcpy( newArray, m_hostArray.data(), m_hostArray.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
            }
            else
            {
                // Copy old elements (device to device).
                if( devArrayLength > 0 )
                {
                    DEMAND_CUDA_CHECK( cudaMemcpy( newArray, oldArray, devArrayLength * sizeof( T ), cudaMemcpyDeviceToDevice ) );
                }
                // Copy new elements (host to device).
                DEMAND_CUDA_CHECK( cudaMemcpy( newArray + devArrayLength, &m_hostArray[devArrayLength],
                                               ( m_hostArray.size() - devArrayLength ) * sizeof( T ), cudaMemcpyHostToDevice ) );
            }
            DEMAND_CUDA_CHECK( cudaFree( oldArray ) );
            m_devArrayLengths[deviceIndex] = static_cast<unsigned int>( m_hostArray.size() );
            return newArray;
        }
        else
        {
            DEMAND_ASSERT( !m_isDirty && !hasGrown );
            return m_devArrays[deviceIndex];
        }
    }

    /// Get the array pointer for the specified device.  Valid only after calling synchronize().
    T* getDeviceArray( unsigned int deviceIndex ) const
    {
        DEMAND_ASSERT( deviceIndex < m_devArrays.size() );
        return m_devArrays[deviceIndex];
    }

  protected:
    std::vector<T>            m_hostArray;
    std::vector<T*>           m_devArrays;
    std::vector<unsigned int> m_devArrayLengths;
    bool                      m_isDirty = true;
};

}  // namespace demandLoading
