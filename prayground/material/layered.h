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
        Layered(std::vector<std::shared_ptr<Material>> materials)
            : m_materials(materials) {}

        void addTopLayer(const std::shared_ptr<Material>& material) {
            m_materials.push_back(material);
        }

        void addBottomLayer(const std::shared_ptr<Material>& material) {
            m_materials.insert(m_materials.begin(), material);
        }

    private:
        std::vector<std::shared_ptr<Material>> m_materials;
#endif
    };

} // namespace prayground