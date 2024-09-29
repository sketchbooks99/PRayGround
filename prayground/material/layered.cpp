#include "layered.h"
#include <prayground/core/cudabuffer.h>

namespace prayground {
    Layered::Layered(const SurfaceCallableID& surface_callable_id, std::vector<std::shared_ptr<Material>> materials)
        : Material(surface_callable_id),
        m_materials(materials)
    {
    }

    void Layered::addTopLayer(const std::shared_ptr<Material>& material)
    {
        m_materials.insert(m_materials.begin(), material);
    }
    void Layered::addBottomLayer(const std::shared_ptr<Material>& material)
    {
        m_materials.push_back(material);
    }

    SurfaceType Layered::surfaceType() const
    {
        return SurfaceType::Layered;
    }

    void Layered::copyToDevice()
    {
        Material::copyToDevice();

        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));

        std::vector<SurfaceInfo> surface_infos;
        surface_infos.push_back(SurfaceInfo{
            .data = d_data,
            .callable_id = surfaceCallableID(),
            .type = surfaceType(),
            .use_bumpmap = useBumpmap(),
            .bumpmap = bumpmapData()
            });
        for (auto& material : m_materials)
        {
            material->copyToDevice();
            SurfaceInfo surface_info{
                .data = material->devicePtr(),
                .callable_id = material->surfaceCallableID(),
                .type = material->surfaceType(),
                .use_bumpmap = material->useBumpmap(),
                .bumpmap = material->bumpmapData()
            };
            surface_infos.push_back(surface_info);
        }
        CUDABuffer<SurfaceInfo> d_surface_infos;
        d_surface_infos.copyToDevice(surface_infos);
        this->d_surface_info = d_surface_infos.deviceData();
    }

    void Layered::setLayerAt(const uint32_t& index, const std::shared_ptr<Material>& material)
    {
        if (index >= m_materials.size()) {
            PG_LOG_FATAL("Tried to set layer, but index out of range");
            return;
        }
        m_materials[index] = material;
    }

    std::shared_ptr<Material> Layered::layerAt(const uint32_t& index) const
    {
        if (index < m_materials.size())
            return m_materials[index];
        else {
            PG_LOG_FATAL("Tried to get layer, but index out of range");
            return nullptr;
        }
    }

    void Layered::free()
    {
        for (auto& material : m_materials)
            material->free();
    }

    Layered::Data Layered::getData() const
    {
        return Data{ static_cast<uint32_t>(m_materials.size()) };
    }
} // namespace prayground