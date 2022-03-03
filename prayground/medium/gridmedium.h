#pragma once 

#include <prayground/core/shape.h>
#include <prayground/core/aabb.h>
#include <prayground/ext/nanovdb/NanoVDB.h>

#ifndef __CUDACC__
#include <prayground/core/file_util.h>
#include <prayground/ext/nanovdb/util/GridHandle.h>
#include <prayground/ext/nanovdb/util/IO.h>
#endif // __CUDACC__

namespace prayground {

    /* 3D volume medium using OpenVDB format */

    /// @todo Initliaze with a empty grid to enable procedual GridMedium calculation
    template <typename T>
    class GridMedium_ : public Shape {
    public:
        struct Data {
            T sigma_a;
            T sigma_s;
            T sigma_t;
            float g;
            void* density;
        };

#ifndef __CUDACC__
        GridMedium_(const std::filesystem::path& filename,
            const T& sigma_a, const T& sigma_s, float g)
            : m_sigma_a(sigma_a), m_sigma_s(sigma_s), m_sigma_t(sigma_a + sigma_s), m_g(g)
        {
            load(filename);
        }

        void load(const std::filesystem::path& filename)
        {
            /* Check if the file exists and file format is correct. */
            auto filepath = pgFindDataPath(filename);
            ASSERT(filepath, "The VDB file '" + filename.string() + "' is not found.");

            auto list = nanovdb::io::readGridMetaData(filepath.value().string());
            pgLog("Loading VDB file '" + filepath.value().string() + " ...");
            for (auto& m : list)
                pgLog("       ", m.gridName);
            ASSERT(list.size() > 0, "The grid data is not found or incorrect.");

            /* Create grid */
            nanovdb::GridHandle<> grid_handle;

            std::string first_gridname = list[0].gridName;

            if (first_gridname.length() > 0)
                grid_handle = nanovdb::io::readGrid<>(filepath.value().string(), first_gridname);
            else
                grid_handle = nanovdb::io::readGrid<>(filepath.value().string());

            if (!grid_handle)
            {
                std::stringstream ss;
                ss << "Unable to read " << first_gridname << " from " << filepath.value().string();
                THROW(ss.str());
            }

            auto* meta_data = m_handle.gridMetaData();
            if (meta_data->isPointData())
                THROW("NanoVDB Point Data cannot be handled by PRayGround.");
            if (meta_data->isLevelSet())
                THROW("NanoVDB Level Sets cannot be handled by PRayGround.");

            ASSERT(m_handle.size() != 0, "The size of grid data is zero.");
        }

        constexpr ShapeType type() override
        {
            return ShapeType::Custom;
        }

        void copyToDevice() override
        {
            Data data;

            if (d_density) 
                CUDA_CHECK(cudaMalloc(&d_density, m_handle.size()));
            CUDA_CHECK(cudaMemcpy(d_density, m_handle.data(), m_handle.size(), cudaMemcpyHostToDevice));

            data = Data{
                .sigma_a = m_sigma_a,
                .sigma_s = m_sigma_s,
                .sigma_t = m_sigma_t,
                .g = m_g,
                .density = d_density
            };
        }

        void free() override
        {
            if (d_density) CUDA_CHECK(cudaFree(d_density));
            Shape::free();
        }

        AABB bound() const override
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

            return AABB{ min, max };
        }

        OptixBuildInput createBuildInput() override
        {
            if (d_aabb_buffer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb_buffer)));
            return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
        }
    private:
        T m_sigma_a, m_sigma_s, m_sigma_t;
        float m_g;

        nanovdb::GridHandle<> m_handle;

        CUdeviceptr d_aabb_buffer;
        void* d_density;
#endif
    };

} // namespace prayground