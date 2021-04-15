#pragma once 

#include "../core/material.h"
#include "../core/bsdf.h"
#include "../texture/constant.h"

namespace oprt {

struct MMAPsData {
    void* texdata;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__
class MMAPs final : public Material {
public:
    explicit MMAPs(const float3& a) : m_texture(new ConstantTexture(a)) {}
    explicit MMAPs(Texture* texture) : m_texture(texture) {}
    ~MMAPs(){}

    void prepare_data() override {
        m_texture->prepare_data();

        MMAPsData data = {
            reinterpret_cast<void*>(m_texture->get_dptr()),
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count)
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(MMAPsData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data), 
            &data, sizeof(MMAPsData), 
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::MMAPs; }
private:
    Texture* m_texture;
};

#else 
CALLABLE_FUNC void CC_FUNC(sample_mmaps)(SurfaceInteraction* si, void* matdata)
{
    const MMAPsData* mmaps = reinterpret_cast<MMAPsData*>(matdata);
    si->attenuation = optixDirectCall<float3, SurfaceInteraction*, void*>(
        mmaps->tex_func_idx, si, mmaps->texdata);
    si->wo = retro_transmit(si->wi, si->n);
    si->trace_terminate = false;
    si->emission = make_float3(0.0f);
}
#endif

}