#pragma once

#include <prayground/core/material.h>

namespace prayground {

  template <typename DataT>
  class CustomMaterial : public Material {
  public:
    using Data = DataT;

#ifndef __CUDACC__
    CustomMaterial(
      const SurfaceCallableID& surface_callable_id, 
      const SurfaceType& surface_type,
      const DataT& data) 
      : Material(surface_callable_id), m_surface_type(surface_type), m_data(data) 
    {}

    ~CustomMaterial() {}

    SurfaceType surfaceType() const override { return m_surface_type; }
    SurfaceInfo surfaceInfo() const override 
    {
      ASSERT(d_data, "Material data on device hasn't been allocated yet.");

      return SurfaceInfo {
        .data = d_data, 
        .callable_id = m_surface_callable_id, 
        .type = SurfaceType::RoughReflection
      };
    }

    void copyToDevice() override
    {
        Material::copyToDevice();
        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(DataT)));
        CUDA_CHECK(cudaMemcpy(
            d_data, 
            &m_data, sizeof(DataT), 
            cudaMemcpyHostToDevice
        ));

        // Copy surface info
        SurfaceInfo surface_info{
          .data = d_data,
          .callable_id = m_surface_callable_id,
          .type = surfaceType()
        };
        if (!d_surface_info)
            CUDA_CHECK(cudaMalloc(&d_surface_info, sizeof(SurfaceInfo)));
        CUDA_CHECK(cudaMemcpy(
            d_surface_info,
            &surface_info, sizeof(SurfaceInfo),
            cudaMemcpyHostToDevice
        ));
    }

    void free() override 
    {
      Material::free();
    } 

    void setData(const DataT& data)
    {
      m_data = data;
    }

    DataT data() const { 
      return m_data; 
    }
    
    // Dummy override
    void setTexture(const std::shared_ptr<Texture>& texture) override {}
    std::shared_ptr<Texture> texture() const override { return nullptr; }

  private:
    SurfaceType m_surface_type;
    DataT m_data;

#endif // __CUDACC__
  };

} // namespace prayground