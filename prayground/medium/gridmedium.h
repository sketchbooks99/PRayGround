#pragma once 

#include <prayground/core/shape.h>
#include <prayground/core/aabb.h>
#include <prayground/ext/nanovdb/NanoVDB.h>

#ifndef __CUDACC__
#include <prayground/core/file_util.h>
#include <prayground/core/load3d.h>
#include <prayground/ext/nanovdb/util/GridHandle.h>
#include <prayground/ext/nanovdb/util/IO.h>
#endif // __CUDACC__

namespace prayground {

    /* 3D volume medium using OpenVDB format */

    /// @todo Initliaze with a empty grid to enable procedual GridMedium calculation
    template <typename Spectrum>
    class GridMedium_ : public Shape {
    public:
        struct Data {
            Spectrum sigma_a;
            Spectrum sigma_s;
            Spectrum sigma_t;
            float g;
            void* density;
        };

#ifndef __CUDACC__
        GridMedium_(const std::filesystem::path& filename,
            const Spectrum& sigma_a, const Spectrum& sigma_s, float g)
            : m_sigma_a(sigma_a), m_sigma_s(sigma_s), m_sigma_t(sigma_a + sigma_s), m_g(g)
        {
            load(filename);
        }

        void load(const std::filesystem::path& filename)
        {
            /* Check if the file exists and file format is correct. */
            auto filepath = pgFindDataPath(filename);
            ASSERT(filepath, "The NanoVDB file '" + filename.string() + "' is not found.");
            
            auto ext = pgGetExtension(filename);
            if (ext == ".nvdb")
            {
                pgLog("Loading NanoVDB file '" + filepath.value().string() + " ...");
                loadNanoVDB(filepath.value(), m_handle);
            }
            else 
                THROW("GridMedium can only load NanoVDB file");
        }

        constexpr ShapeType type() override
        {
            return ShapeType::Custom;
        }

        void copyToDevice() override
        {
            auto data = this->getData();

            // Copy data to device through Shape::d_data
            if (!d_data)
                CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
            CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
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

        Data getData() 
        {
            // Copy density data to device
            if (!d_density) 
                CUDA_CHECK(cudaMalloc(&d_density, m_handle.size()));
            CUDA_CHECK(cudaMemcpy(d_density, m_handle.data(), m_handle.size(), cudaMemcpyHostToDevice));

            return Data{
                .sigma_a = m_sigma_a,
                .sigma_s = m_sigma_s,
                .sigma_t = m_sigma_t,
                .g = m_g,
                .density = d_density
            };
        }
    private:
        Spectrum m_sigma_a, m_sigma_s, m_sigma_t;
        float m_g;

        nanovdb::GridHandle<> m_handle;

        CUdeviceptr d_aabb_buffer{ 0 };
        void* d_density{ nullptr };
#endif
    };

} // namespace prayground