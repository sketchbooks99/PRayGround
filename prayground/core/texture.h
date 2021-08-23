#pragma once 

#include <prayground/core/util.h>
#include <sutil/vec_math.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace prayground {

#ifndef __CUDACC__

class Texture {
public:
    // Preparing texture data on the device.
    virtual void copyToDevice() = 0;
    virtual void free()
    {
        if (d_data) cuda_free(d_data);
    }

    // Get data pointer on the device.
    void* devicePtr() const 
    {
        return d_data;
    }

    void setProgramId(const int prg_id)
    {
        m_prg_id = prg_id;
    }
    int programId() const 
    {
        return m_prg_id;
    }
protected:
    // CUdeviceptr d_data { 0 };
    void* d_data;
    int m_prg_id { -1 };
};

#endif

} // ::prayground
