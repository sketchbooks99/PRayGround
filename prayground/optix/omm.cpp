#include "omm.h"

namespace prayground {
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

    void OpacityMicroMap::build(const Context& ctx, const std::shared_ptr<Texture>& opacity_texture)
    {
        ASSERT(m_mesh_ptr != nullptr, "Mesh to construct OMM hasn't set");
        ASSERT(!m_mesh_ptr->texcoords().empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int N_TRIANGLES = m_mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        // Allocate opacity buffer
        unsigned short** omm_opacity_data = new unsigned short* [N_TRIANGLES];
        size_t N_MICROMAP_PER_BYTE = (N_MICRO_TRIANGLES / sizeof(unsigned short)) / m_settings.format;
        for (size_t i = 0; i < N_TRIANGLES; i++)
            omm_opacity_data[i] = new unsigned short[N_MICROMAP_PER_BYTE];

        // Determine opacity map according to texture contains alpha value


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
                .subdivisionLevel = static_cast<unsigned short>(m_settings.subdivision_level),
                .format = static_cast<unsigned short>(m_settings.format)
            };
            offset += sizeof(unsigned short) * N_MICROMAP_PER_BYTE;
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

    void OpacityMicroMap::build(const Context& ctx, const std::function<int(const Vec2f&, const Vec2f&, const Vec2f&, const Vec2f*)> opacity_func)
    {

    }

    OptixBuildInputOpacityMicromap OpacityMicroMap::getBuildInputForGAS() const
    {
        return OptixBuildInputOpacityMicromap();
    }
} // namespace prayground