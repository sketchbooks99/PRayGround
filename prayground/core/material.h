#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/core/interaction.h>
#include <prayground/core/util.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace prayground {

// Abstract class to compute scattering properties.
class Material {
/// @note Make this class be dummy class on device kernels
#ifndef __CUDACC__

public:
    virtual ~Material() {}

    virtual SurfaceType surfaceType() const = 0;
    
    virtual void copyToDevice() = 0;

    virtual void free()
    {
        if (d_data)
        {
            CUDA_CHECK(cudaFree(d_data));
            d_data = nullptr;
        }
    }
    
    void* devicePtr() const { return d_data; }
protected:
    void* d_data { nullptr };
#endif
};

} // ::prayground