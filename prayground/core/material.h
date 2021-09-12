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

#ifndef __CUDACC__

// Abstract class to compute scattering properties.
class Material {
public:
    virtual ~Material() {}

    virtual SurfaceType surfaceType() const = 0;
    
    virtual void copyToDevice() = 0;

    virtual void free()
    {
        if (d_data) cuda_free(d_data);
    }
    
    void* devicePtr() const { return d_data; }
protected:
    void* d_data { 0 };
};

#endif

} // ::prayground