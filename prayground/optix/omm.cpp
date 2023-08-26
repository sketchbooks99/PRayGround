#include "omm.h"
#include <optix_micromap.h>
#include <numeric>
#include <prayground/texture/bitmap.h>
#include <prayground/core/cudabuffer.h>

namespace prayground {
    static void constructOpacitymap(
        uint16_t** out_opacity_map, 
        const std::vector<Vec2f>& texcoords,
        const std::vector<Vec3i>& indices,
        const OpacityMicromap::Settings &settings, 
        const std::function<int(const OpacityMicromap::MicroBarycentrics&, const Vec2f&, const Vec2f&, const Vec2f&)>& eval_func
    )
    {
        const int N_TRIANGLES = indices.size();
        const int N_MICRO_TRIANGLES = 1 << (settings.subdivision_level * 2);
        const size_t N_STATES_PER_ELEM = 16 / settings.format;

        for (int i = 0; const auto& index : indices)
        {
            const Vec2f uv0 = texcoords[index.x()];
            const Vec2f uv1 = texcoords[index.y()];
            const Vec2f uv2 = texcoords[index.z()];

            for (uint32_t j = 0; j < N_MICRO_TRIANGLES; j++)
            { 
                // Barycentric coordinates at micro triangle
                float2 bary0, bary1, bary2;
                optixMicromapIndexToBaseBarycentrics(j, settings.subdivision_level, bary0, bary1, bary2);
                
                auto barycentrics = OpacityMicromap::MicroBarycentrics{ bary0, bary1, bary2 };
                auto state = eval_func(barycentrics, uv0, uv1, uv2);

                out_opacity_map[i][j / N_STATES_PER_ELEM] |= state << (j % N_STATES_PER_ELEM * settings.format);
            }
            i++;
        }
    }

    OpacityMicromap::OpacityMicromap()
        : m_buffers{{}}
    {
        ASSERT(OPTIX_VERSION >= 70600, "Opacity micro map requires OptiX 7.6 at the minimum");
        m_settings.build_flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
        m_settings.format = OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE;
        m_settings.subdivision_level = 3; // 4^3 = 64 micro-triangles per single triangle
    }

    OpacityMicromap::OpacityMicromap(const Settings& settings)
        : m_settings(settings), m_buffers{{}}
    {
        ASSERT(OPTIX_VERSION >= 70600, "Opacity micro map requires OptiX 7.6 at the minimum");
    }

    void OpacityMicromap::buildFromBitmapTexture(
        const Context& ctx, 
        const std::vector<Vec2f>& texcoords, 
        const std::vector<Vec3i>& indices,
        const std::shared_ptr<BitmapTexture>& opacity_texture)
    {
        ASSERT(!texcoords.empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int m_num_triangles = indices.size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        uint16_t** omm_opacity_data = new uint16_t * [m_num_triangles];
        size_t N_STATES_PER_ELEM = 16 / m_settings.format;
        for (size_t i = 0; i < m_num_triangles; i++)
            omm_opacity_data[i] = new uint16_t[N_MICRO_TRIANGLES / N_STATES_PER_ELEM];

        int width = opacity_texture->width();
        int height = opacity_texture->height();

        auto evalateOpacity = [&](const OpacityMicromap::MicroBarycentrics& mbc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2) -> int
        {
            const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, mbc.uv0);
            const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, mbc.uv1);
            const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, mbc.uv2);

            Vec4u pixel0 = opacity_texture->eval(bary0);
            Vec4u pixel1 = opacity_texture->eval(bary1);
            Vec4u pixel2 = opacity_texture->eval(bary2);

            if (pixel0.w() == 0 && pixel1.w() == 0 && pixel2.w() == 0)
                return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
            if (pixel0.w() != 0 && pixel1.w() != 0 && pixel2.w() != 0)
                return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            else 
            {
                if (m_settings.format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                else // Treat micro triangle as opaque when the state is controlled with 1 bit (0 or 1)
                    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            }
        };

        constructOpacitymap(omm_opacity_data, texcoords, indices, m_settings, evalateOpacity);

        buildFromOpacitymap(omm_opacity_data, ctx);
    }

    void OpacityMicromap::buildFromBitmapTexture(
        const Context& ctx,
        const std::vector<Vec2f>& texcoords,
        const std::vector<Vec3i>& indices,
        const std::shared_ptr<FloatBitmapTexture>& opacity_texture
    )
    {
        ASSERT(!texcoords.empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int m_num_triangles = indices.size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        uint16_t** omm_opacity_data = new uint16_t * [m_num_triangles];
        size_t N_STATES_PER_ELEM = 16 / m_settings.format;
        for (size_t i = 0; i < m_num_triangles; i++)
            omm_opacity_data[i] = new uint16_t[N_MICRO_TRIANGLES / N_STATES_PER_ELEM];

        int width = opacity_texture->width();
        int height = opacity_texture->height();

        auto evalateOpacity = [&](const OpacityMicromap::MicroBarycentrics& mbc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2) -> int
        {
            const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, mbc.uv0);
            const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, mbc.uv1);
            const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, mbc.uv2);

            Vec4f pixel0 = opacity_texture->eval(bary0);
            Vec4f pixel1 = opacity_texture->eval(bary1);
            Vec4f pixel2 = opacity_texture->eval(bary2);

            if (pixel0.w() == 0 && pixel1.w() == 0 && pixel2.w() == 0)
                return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
            if (pixel0.w() != 0 && pixel1.w() != 0 && pixel2.w() != 0)
                return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            else
            {
                if (m_settings.format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                else // Treat micro triangle as opaque when the state is controlled with 1 bit (0 or 1)
                    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            }
        };

        constructOpacitymap(omm_opacity_data, texcoords, indices, m_settings, evalateOpacity);

        buildFromOpacitymap(omm_opacity_data, ctx);
    }

    void OpacityMicromap::build(
        const Context& ctx, 
        const std::vector<Vec2f>& texcoords,
        const std::vector<Vec3i>& indices,
        const std::function<int(const OpacityMicromap::MicroBarycentrics&, const Vec2f&, const Vec2f&, const Vec2f&)>& opacity_func)
    {
        ASSERT(!texcoords.empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        m_num_triangles = indices.size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        uint16_t** omm_opacity_data = new uint16_t * [m_num_triangles];
        size_t N_STATES_PER_ELEM = 16 / m_settings.format;
        for (size_t i = 0; i < m_num_triangles; i++)
            omm_opacity_data[i] = new uint16_t[N_MICRO_TRIANGLES / N_STATES_PER_ELEM];

        constructOpacitymap(omm_opacity_data, texcoords, indices, m_settings, opacity_func);
        buildFromOpacitymap(omm_opacity_data, ctx);
    }

    OptixBuildInputOpacityMicromap OpacityMicromap::getBuildInputForGAS() const
    {
        // Check if OMM has already been builded
        ASSERT(m_buffers.output != 0, "OMM has not been builded yet.");

        OptixOpacityMicromapUsageCount usage_count = {};
        usage_count.count = static_cast<uint32_t>(m_num_triangles);
        usage_count.format = m_settings.format;
        usage_count.subdivisionLevel = m_settings.subdivision_level;

        // Prepare OMM indices 
        std::vector<uint16_t> omm_indices(m_num_triangles);
        std::iota(omm_indices.begin(), omm_indices.end(), 1);

        CUDABuffer<uint16_t> d_omm_indices;
        d_omm_indices.copyToDevice(omm_indices);

        OptixBuildInputOpacityMicromap omm_input = {};
        omm_input.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
        omm_input.opacityMicromapArray = m_buffers.output;
        omm_input.indexBuffer = d_omm_indices.devicePtr();
        omm_input.indexSizeInBytes = sizeof(uint16_t);
        omm_input.numMicromapUsageCounts = 1;
        omm_input.micromapUsageCounts = &usage_count;

        return omm_input;
    }

    void OpacityMicromap::buildFromOpacitymap(uint16_t** omm_opacity_data, const Context& ctx)
    {
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);
        const int N_STATES_PER_ELEM = 16 / m_settings.format;

        // Copy opacity buffer to device
        CUdeviceptr d_omm_opacity_data = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_omm_opacity_data), sizeof(omm_opacity_data)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_omm_opacity_data), omm_opacity_data, sizeof(omm_opacity_data), cudaMemcpyHostToDevice));

        // Build OMM
        OptixOpacityMicromapHistogramEntry histogram{};
        histogram.count = m_num_triangles;
        histogram.format = m_settings.format;
        histogram.subdivisionLevel = m_settings.subdivision_level;

        // Currently, only single histogram entry is allowed
        OptixOpacityMicromapArrayBuildInput build_input{};
        build_input.flags = m_settings.build_flags;
        build_input.inputBuffer = d_omm_opacity_data;
        build_input.numMicromapHistogramEntries = 1;
        build_input.micromapHistogramEntries = &histogram;

        // Calculate memory usage for OMM
        OptixMicromapBufferSizes buffer_sizes{};
        OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(static_cast<OptixDeviceContext>(ctx), &build_input, &buffer_sizes));

        // Setup descriptor
        std::vector<OptixOpacityMicromapDesc> omm_descs(m_num_triangles);
        uint32_t offset = 0;
        for (auto& desc : omm_descs)
        {
            desc = {
                .byteOffset = offset,
                .subdivisionLevel = static_cast<uint16_t>(m_settings.subdivision_level),
                .format = static_cast<uint16_t>(m_settings.format)
            };
            offset += sizeof(uint16_t) * (N_MICRO_TRIANGLES / N_STATES_PER_ELEM);
        }

        CUDABuffer<OptixOpacityMicromapDesc> d_omm_descs;
        d_omm_descs.copyToDevice(omm_descs);

        build_input.perMicromapDescBuffer = d_omm_descs.devicePtr();
        build_input.perMicromapDescStrideInBytes = 0;

        CUdeviceptr d_temp_buffer = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_buffers.output), buffer_sizes.outputSizeInBytes));

        OptixMicromapBuffers micromap_buffers{};
        micromap_buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
        micromap_buffers.temp = d_temp_buffer;
        micromap_buffers.tempSizeInBytes = buffer_sizes.tempSizeInBytes;

        OPTIX_CHECK(optixOpacityMicromapArrayBuild(static_cast<OptixDeviceContext>(ctx), 0, &build_input, &micromap_buffers));

        // Free buffers
        cuda_frees(d_omm_opacity_data, d_temp_buffer);
        d_omm_descs.free();
    }
} // namespace prayground