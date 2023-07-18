#pragma once

#ifndef __CUDACC__
#include <prayground/optix/macros.h>
#endif

namespace prayground {

    enum class EmitterType {
        Area = 0, 
        Point = 1,
        Envmap = 2,
        Count = 3
    };

    class Emitter {

#ifndef __CUDACC__
    public:
        virtual void copyToDevice() = 0;
        virtual EmitterType type() const = 0;

        virtual void free() 
        {
            if (d_data)
                CUDA_CHECK(cudaFree(d_data));
        }

        void* devicePtr() const { return reinterpret_cast<void*>(d_data); }
    protected:
        void* d_data { nullptr };
#endif
    };

} // namespace prayground
