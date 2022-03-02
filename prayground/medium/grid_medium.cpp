#include "grid_medium.h"
#include <prayground/core/file_util.h>
#include <prayground/ext/nanovdb/util/IO.h>

namespace prayground {

    Volume::Volume(const std::filesystem::path& filepath)
    {
        load(filepath);
    }

    void Volume::load(const std::filesystem::path& filepath)
    {
        /* Check if the file exists and file format is correct. */
        auto ext = pgGetExtension(filepath);
        ASSERT(ext == ".vdb", "Only OpenVDB file is loaded.");

        ASSERT(std::filesystem::exists(filepath), "The VDB file '" + filepath.string() + "' is not found.");

        auto list = nanovdb::io::readGridMetaData(filepath.string());
        pgLog("Loading VDB file '" + filepath.string() + " ...");
        for (auto& m : list)
            pgLog("       ", m.gridName);
        ASSERT(list.size() > 0, "The grid data is not found or incorrect.");

        /* Create grid */
        nanovdb::GridHandle<> grid_handle;

        std::string first_gridname = list[0].gridName;

        if (first_gridname.length() > 0)
            grid_handle = nanovdb::io::readGrid<>(filepath.string(), first_gridname);
        else
            grid_handle = nanovdb::io::readGrid<>(filepath.string());

        if (!grid_handle)
        {
            std::stringstream ss;
            ss << "Unable to read " << first_gridname << " from " << filepath.string();
            THROW(ss.str());
        }

        auto* meta_data = m_handle.gridMetaData();
        if (meta_data->isPointData())
            THROW("NanoVDB Point Data cannot be handled by PRayGround.");
        if (meta_data->isLevelSet())
            THROW("NanoVDB Level Sets cannot be handled by PRayGround.");
        
        ASSERT(m_handle.size() != 0, "The size of grid data is zero.");
    }

    constexpr ShapeType Volume::type() 
    {
        return ShapeType::Custom;
    }

    void Volume::copyToDevice() 
    {
        Data data;

        CUDA_CHECK(cudaMalloc(&data.grid, m_handle.size()));
        CUDA_CHECK(cudaMemcpy(data.grid, m_handle.data(), m_handle.size(), cudaMemcpyHostToDevice));

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data, &data, sizeof(Data), cudaMemcpyHostToDevice
        ));
    }

    void Volume::free()
    {
        Shape::free();
    }

    AABB Volume::bound() const 
    {
        auto grid_handle = m_handle.grid<float>();

        auto bbox = grid_handle->indexBBox();
        nanovdb::Coord bounds_min(bbox.min());
        nanovdb::Coord bounds_max(bbox.max() + nanovdb::Coord(1));

        float3 min = {
            static_cast<float>(bounds_min[0]), 
            static_cast<float>(bounds_min[1]), 
            static_cast<float>(bounds_min[2]) 
        };
        float3 max = {
            static_cast<float>(bounds_max[0]),
            static_cast<float>(bounds_max[1]),
            static_cast<float>(bounds_max[2])
        };

        return AABB{min, max};        
    }

    OptixBuildInput Volume::createBuildInput()
    {
        if (d_aabb_buffer) cuda_free(d_aabb_buffer);
        createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
    }

} // namespace prayground