#pragma once 

#include <prayground/core/shape.h>
#include <prayground/core/aabb.h>
#include <prayground/ext/nanovdb/NanoVDB.h>
#include <prayground/ext/nanovdb/util/GridHandle.h>
#include <filesystem>

namespace prayground {

    /// @todo Initliaze with a empty grid to enable procedual volume calculation
    class Volume : public Shape {
    public:
        struct Data {
            void* grid;
        };

#ifndef __CUDACC__
        Volume(const std::filesystem::path& filepath);

        void load(const std::filesystem::path& filepath);

        constexpr ShapeType type() override;

        void copyToDevice() override;
        void free() override;

        AABB bound() const override;

        OptixBuildInput createBuildInput() override;

    private:
        nanovdb::GridHandle<> m_handle;
        CUdeviceptr d_aabb_buffer;

#endif
    };

} // namespace prayground