#include "omm.h"
#include <optix_micromap.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>
#include <prayground/texture/bitmap.h>

namespace prayground {
    static void constructOpacitymap(
        uint16_t** out_opacity_map, 
        const std::shared_ptr<TriangleMesh>& mesh_ptr,  
        const OpacityMicroMap::Settings &settings, 
        const std::function<int(const OpacityMicroMap::MicroBarycentrics&, const Vec2f&, const Vec2f&, const Vec2f&)>& eval_func
    )
    {
        const int N_TRIANGLES = mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (settings.subdivision_level * 2);
        const size_t N_STATES_PER_ELEM = 16 / settings.format;

        for (int i = 0; const auto& face : mesh_ptr->faces())
        {
            const Vec2f uv0 = mesh_ptr->texcoordAt(face.texcoord_id.x());
            const Vec2f uv1 = mesh_ptr->texcoordAt(face.texcoord_id.x());
            const Vec2f uv2 = mesh_ptr->texcoordAt(face.texcoord_id.z());

            for (uint32_t j = 0; j < N_MICRO_TRIANGLES; j++)
            { 
                // Barycentric coordinates at micro triangle
                float2 bary0, bary1, bary2;
                optixMicromapIndexToBaseBarycentrics(j, settings.subdivision_level, bary0, bary1, bary2);
                
                auto barycentrics = OpacityMicroMap::MicroBarycentrics{ bary0, bary1, bary2 };
                auto state = eval_func(barycentrics, uv0, uv1, uv2);

                out_opacity_map[i][j / N_STATES_PER_ELEM] |= state << (j % N_STATES_PER_ELEM * settings.format);
            }
            i++;
        }
    }

    OpacityMicroMap::OpacityMicroMap()
        : m_buffers{{}}
    {
        m_settings.build_flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
        m_settings.format = OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE;
        m_settings.subdivision_level = 3; // 4^3 = 64 micro-triangles per single triangle
    }

    OpacityMicroMap::OpacityMicroMap(const Settings& settings)
        : m_settings(settings), m_buffers{{}}
    {

    }

    OpacityMicroMap::OpacityMicroMap(const Settings& settings, const std::shared_ptr<TriangleMesh>& mesh)
        : m_settings{settings}, m_buffers{{}}, m_mesh_ptr{mesh}
    {
    }

    void OpacityMicroMap::setMesh(const std::shared_ptr<TriangleMesh>& mesh)
    {
        m_mesh_ptr = mesh;
    }

    void OpacityMicroMap::buildFromBitmapTexture(const Context& ctx, const std::shared_ptr<BitmapTexture>& opacity_texture)
    {
        ASSERT(m_mesh_ptr != nullptr, "Mesh to construct OMM hasn't set");
        ASSERT(!m_mesh_ptr->texcoords().empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int N_TRIANGLES = m_mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        uint16_t** omm_opacity_data = new uint16_t * [N_TRIANGLES];
        size_t N_STATES_PER_ELEM = 16 / m_settings.format;
        for (size_t i = 0; i < N_TRIANGLES; i++)
            omm_opacity_data[i] = new uint16_t[N_MICRO_TRIANGLES / N_STATES_PER_ELEM];

        int width = opacity_texture->width();
        int height = opacity_texture->height();

        auto evalateOpacity = [&](const OpacityMicroMap::MicroBarycentrics& mbc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2) -> int
        {
            const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, mbc.uv0);
            const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, mbc.uv1);
            const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, mbc.uv2);

            const Vec2i idx0{ (int)(bary0.x() * width), (int)(bary0.y() * height) };
            const Vec2i idx1{ (int)(bary1.x() * width), (int)(bary1.y() * height) };
            const Vec2i idx2{ (int)(bary2.x() * width), (int)(bary2.y() * height) };

            Vec4u pixel0 = std::get<Vec4u>(opacity_texture->at(idx0.x(), idx0.y()));
            Vec4u pixel1 = std::get<Vec4u>(opacity_texture->at(idx1.x(), idx1.y()));
            Vec4u pixel2 = std::get<Vec4u>(opacity_texture->at(idx2.x(), idx2.y()));

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

        constructOpacitymap(omm_opacity_data, m_mesh_ptr, m_settings, evalateOpacity);

        buildFromOpacitymap(omm_opacity_data, ctx);
    }

    void OpacityMicroMap::buildFromBitmapTexture(const Context& ctx, const std::shared_ptr<FloatBitmapTexture>& opacity_texture)
    {
        ASSERT(m_mesh_ptr != nullptr, "Mesh to construct OMM hasn't set");
        ASSERT(!m_mesh_ptr->texcoords().empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int N_TRIANGLES = m_mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        uint16_t** omm_opacity_data = new uint16_t * [N_TRIANGLES];
        size_t N_STATES_PER_ELEM = 16 / m_settings.format;
        for (size_t i = 0; i < N_TRIANGLES; i++)
            omm_opacity_data[i] = new uint16_t[N_MICRO_TRIANGLES / N_STATES_PER_ELEM];

        int width = opacity_texture->width();
        int height = opacity_texture->height();

        auto evalateOpacity = [&](const OpacityMicroMap::MicroBarycentrics& mbc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2) -> int
        {
            const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, mbc.uv0);
            const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, mbc.uv1);
            const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, mbc.uv2);

            const Vec2i idx0{ (int)(bary0.x() * width), (int)(bary0.y() * height) };
            const Vec2i idx1{ (int)(bary1.x() * width), (int)(bary1.y() * height) };
            const Vec2i idx2{ (int)(bary2.x() * width), (int)(bary2.y() * height) };

            Vec4f pixel0 = std::get<Vec4f>(opacity_texture->at(idx0.x(), idx0.y()));
            Vec4f pixel1 = std::get<Vec4f>(opacity_texture->at(idx1.x(), idx1.y()));
            Vec4f pixel2 = std::get<Vec4f>(opacity_texture->at(idx2.x(), idx2.y()));

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

        constructOpacitymap(omm_opacity_data, m_mesh_ptr, m_settings, evalateOpacity);

        buildFromOpacitymap(omm_opacity_data, ctx);
    }

    void OpacityMicroMap::build(const Context& ctx, const std::function<int(const OpacityMicroMap::MicroBarycentrics&, const Vec2f&, const Vec2f&, const Vec2f&)>& opacity_func)
    {
        ASSERT(m_mesh_ptr != nullptr, "Mesh to construct OMM hasn't set");
        ASSERT(!m_mesh_ptr->texcoords().empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int N_TRIANGLES = m_mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        uint16_t** omm_opacity_data = new uint16_t * [N_TRIANGLES];
        size_t N_STATES_PER_ELEM = 16 / m_settings.format;
        for (size_t i = 0; i < N_TRIANGLES; i++)
            omm_opacity_data[i] = new uint16_t[N_MICRO_TRIANGLES / N_STATES_PER_ELEM];

        constructOpacitymap(omm_opacity_data, m_mesh_ptr, m_settings, opacity_func);
        buildFromOpacitymap(omm_opacity_data, ctx);
    }

    OptixBuildInputOpacityMicromap OpacityMicroMap::getBuildInputForGAS() const
    {
        return OptixBuildInputOpacityMicromap();
    }
    void OpacityMicroMap::buildFromOpacitymap(uint16_t** omm_opacity_data, const Context& ctx)
    {
        const int N_TRIANGLES = m_mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);
        const int N_STATES_PER_ELEM = 16 / m_settings.format;

        // Copy opacity buffer to device
        CUdeviceptr d_omm_opacity_data = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_omm_opacity_data), sizeof(omm_opacity_data)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_omm_opacity_data), omm_opacity_data, sizeof(omm_opacity_data), cudaMemcpyHostToDevice));

        // Build OMM
        OptixOpacityMicromapHistogramEntry histogram{};
        histogram.count = N_TRIANGLES;
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
        std::vector<OptixOpacityMicromapDesc> omm_descs(N_TRIANGLES);
        uint32_t offset = 0;
        for (auto& desc : omm_descs)
        {
            desc = {
                .byteOffset = offset,
                .subdivisionLevel = static_cast<uint16_t>(m_settings.subdivision_level),
                .format = static_cast<uint16_t>(m_settings.format)
            };
            offset += sizeof(uint16_t) * N_STATES_PER_ELEM;
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

        cuda_frees(d_omm_opacity_data, d_temp_buffer);
        d_omm_descs.free();
    }
} // namespace prayground