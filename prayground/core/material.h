#pragma once

#include <cuda_runtime.h>
#include <prayground/core/interaction.h>
#include <prayground/optix/macros.h>
#include <prayground/core/texture.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace prayground {

    // Abstract class to compute scattering properties.
    class Material {

    public:
        struct Data {
            cudaTextureObject_t bumpmap { 0 };
            int bumpmap_id { -1 };
        };

    /// @note Make this class be dummy class on device kernels
#ifndef __CUDACC__
    public:
        Material(const SurfaceCallableID& surface_callable_id, const std::shared_ptr<Texture>& bumpmap = nullptr, int bumpmap_id = -1)
            : m_surface_callable_id(surface_callable_id), m_bumpmap(bumpmap), m_bumpmap_id(bumpmap_id) {}
        virtual ~Material() {}

        virtual SurfaceType surfaceType() const = 0;

        virtual SurfaceInfo surfaceInfo() const = 0;
    
        virtual void copyToDevice() = 0;

        virtual void setTexture(const std::shared_ptr<Texture>& texture) = 0;
        virtual std::shared_ptr<Texture> texture() const = 0;

        virtual void free()
        {
            if (d_data)
            {
                CUDA_CHECK(cudaFree(d_data));
                d_data = nullptr;
            }
        }


        void setBumpmap(const std::shared_ptr<Texture>& bumpmap) {
            m_bumpmap = bumpmap;
        }

        const SurfaceCallableID& surfaceCallableID() const
        {
            return m_surface_callable_id;
        }
    
        void* devicePtr() const { return d_data; }
    protected:
        SurfaceCallableID m_surface_callable_id;
        void* d_data { nullptr };

        std::shared_ptr<Texture> m_bumpmap;
        int m_bumpmap_id { -1 };

        // TODO: Displacement map
#endif // __CUDACC__
    };

} // namespace prayground