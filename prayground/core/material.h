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
    /// @note Make this class be dummy class on device kernels
#ifndef __CUDACC__
    public:
        Material() = default;
        Material(const SurfaceCallableID& surface_callable_id, const std::shared_ptr<Texture>& bumpmap = nullptr, int bumpmap_id = -1)
            : m_surface_callable_id(surface_callable_id), m_bumpmap(bumpmap), m_bumpmap_id(bumpmap_id) {}
        virtual ~Material() {}

        virtual SurfaceType surfaceType() const = 0;
    
        virtual void copyToDevice() {
            if (m_bumpmap && !m_bumpmap->devicePtr())
                m_bumpmap->copyToDevice();
        }

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
        Texture::Data bumpmapData() const {
            if (m_bumpmap)
                return m_bumpmap->getData();
            else
                return { nullptr, -1 };
        }
        bool useBumpmap() const {
            return m_bumpmap != nullptr;
        }

        const SurfaceCallableID& surfaceCallableID() const
        {
            return m_surface_callable_id;
        }
    
        void* devicePtr() const { return d_data; }

        SurfaceInfo* surfaceInfoDevicePtr() const { return d_surface_info; }


    protected:
        SurfaceCallableID m_surface_callable_id;
        void* d_data { nullptr };

        // Must be copied to the device in Derived::copyToDevice()
        SurfaceInfo* d_surface_info{ nullptr };

        std::shared_ptr<Texture> m_bumpmap{ nullptr };
        int m_bumpmap_id { -1 };

        // TODO: Displacement map
#endif // __CUDACC__
    };

} // namespace prayground