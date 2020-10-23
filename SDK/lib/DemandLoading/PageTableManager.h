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

#include <algorithm>
#include <limits>
#include <vector>

namespace demandLoading {

/// The PageTableManager is used to reserve a contiguous range of page table entries.  It keeps a
/// mapping that allows the resource corresponding to a page table entry to be determined in log(N)
/// time.
class PageTableManager
{
  public:
    explicit PageTableManager( unsigned int totalPages )
        : m_totalPages( totalPages )
    {
    }

    unsigned int getAvailablePages() { return m_totalPages - m_nextPage; }

    unsigned int getHighestUsedPage() { return m_nextPage - 1; }

    /// Reserve the specified number of contiguous page table entries, associating them with the
    /// specified resource id.  Returns the first page reserved.
    unsigned int reserve( unsigned int numPages, unsigned int resourceId )
    {
        DEMAND_ASSERT_MSG( getAvailablePages() >= numPages, "Insufficient pages in optix page table" );
        PageMapping mapping{m_nextPage, m_nextPage + numPages - 1, resourceId};
        m_mappings.push_back( mapping );
        m_nextPage += numPages;
        return mapping.firstPage;
    }

    /// Find the resource associated with the specified page.  Returns UINT_MAX if not found.
    unsigned int getResource( unsigned int pageId )
    {
        // Pages are allocated in increasing order, so the array of mappings is sorted, allowing us
        // to use binary search to find the the given page id.
        auto least =
            std::lower_bound( m_mappings.cbegin(), m_mappings.cend(), pageId,
                              []( const PageMapping& entry, unsigned int pageId ) { return pageId > entry.lastPage; } );
        return ( least != m_mappings.cend() ) ? least->resourceId : std::numeric_limits<unsigned int>::max();
    }

  private:
    struct PageMapping
    {
        unsigned int firstPage;
        unsigned int lastPage;
        unsigned int resourceId;
    };

    unsigned int             m_totalPages;
    unsigned int             m_nextPage = 0;
    std::vector<PageMapping> m_mappings;
};

}  // namespace demandLoading
