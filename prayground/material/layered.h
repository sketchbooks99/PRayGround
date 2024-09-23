/* 
Ref: Arbitrarily Layered Micro-Facet Surfaces [Weidlich and Wilkie, 2007]


*/

#pragma once 

#include <prayground/core/material.h>

namespace prayground {

    class Layered final : public Material {
    public:
        struct Data {
            uint32_t num_layers;
        };
#ifndef __CUDACC__
        /* Upper layers are added to rear, and lower layers are added to front of material vector */
        Layered(std::vector<std::shared_ptr<Material>> materials);

        void addTopLayer(const std::shared_ptr<Material>& material);

        void addBottomLayer(const std::shared_ptr<Material>& material);

        SurfaceType surfaceType() const override;
        void copyToDevice() override;

        void setTexture(const std::shared_ptr<Texture>& texture) override {}
        std::shared_ptr<Texture> texture() const override { return nullptr; }

        void free() override;

        Data getData() const;

    private:
        std::vector<std::shared_ptr<Material>> m_materials;
#endif
    };

} // namespace prayground