#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <prayground/core/util.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace prayground {

#ifndef __CUDACC__

// Abstract class to compute scattering properties.
class Material {
public:
    virtual ~Material() {}
    
    virtual void copyToDevice() = 0;

    virtual void free()
    {
        if (d_data) cuda_free(d_data);
    }
    
    void* devicePtr() const { return d_data; }

    void addProgramId(const uint32_t idx)
    {
        m_prg_ids.push_back(idx);
    }

    int32_t programIdAt(const int32_t idx) const
    {
        if (idx >= m_prg_ids.size())
        {
            Message(MSG_ERROR, "prayground::Material::funcIdAt(): Index to get function id of material exceeds the number of functions");
            return -1;
        }
        return static_cast<int32_t>(m_prg_ids[idx]);
    }
protected:
    void* d_data { 0 };
    std::vector<uint32_t> m_prg_ids;
};

#endif

} // ::prayground