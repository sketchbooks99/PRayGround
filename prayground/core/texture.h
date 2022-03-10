#pragma once 

#include <prayground/core/util.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace prayground {

class Texture {
public:
    struct Data {
        void* data;
        int prg_id;
    };

#ifndef __CUDACC__
    Texture(int prg_id) : m_prg_id(prg_id) {}

    // Preparing texture data on the device.
    virtual void copyToDevice() = 0;
    virtual void free()
    {
        if (d_data) CUDA_CHECK(cudaFree(d_data));
        d_data = nullptr;
    }

    // Get data pointer on the device.
    void* devicePtr() const 
    {
        return d_data;
    }

    Data getData() const
    {
        return
        {
            d_data,
            m_prg_id
        };
    }

    void setProgramId(const uint32_t prg_id)
    {
        m_prg_id = static_cast<int>(prg_id);
    }
    uint32_t programId() const 
    {
        ASSERT(m_prg_id > -1, "Please set program id to texture.");
        return static_cast<uint32_t>(m_prg_id);
    }
#endif

protected:
    void* d_data{ nullptr };
    int m_prg_id;
};

} // ::prayground
