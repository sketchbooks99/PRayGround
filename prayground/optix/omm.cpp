#include "omm.h"
#include <optix_micromap.h>
#include <numeric>
#include <prayground/texture/bitmap.h>
#include <prayground/core/cudabuffer.h>

namespace prayground {
    // ------------------------------------------------------------------
    OpacityMicromap::OpacityMicromap()
        : m_buffers{{}}
    {

    }

    // ------------------------------------------------------------------
    void OpacityMicromap::build(const Context& ctx, CUstream stream, const Input& input, uint32_t build_flags)
    {
        ASSERT(input.texcoords != nullptr, "Incorrect texture coodinate data");
        ASSERT(input.faces != nullptr && input.num_faces > 0, "Incorrect face data");
        ASSERT(input.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int num_micro_triangles = 1 << (input.subdivision_level * 2);

        size_t num_states_per_elem = 16 / input.format;
        size_t num_elems_per_face = (num_micro_triangles / 16 * input.format) + 1;

        std::vector<uint16_t> omm_opacity_data(input.num_faces * num_elems_per_face);

        bool is_bitmap = std::holds_alternative<std::shared_ptr<BitmapTexture>>(input.opacity_bitmap_or_function);
        bool is_fbitmap = std::holds_alternative<std::shared_ptr<FloatBitmapTexture>>(input.opacity_bitmap_or_function);
        bool is_lambda = std::holds_alternative<OpacityMicromap::OpacityFunction>(input.opacity_bitmap_or_function);

        ASSERT(is_bitmap || is_fbitmap || is_lambda, "Invalid bitmap or function to construct opacity micromap");

        for (size_t i = 0; i < input.num_faces; i++) {
            const Vec2f uv0 = input.texcoords[input.faces[i].x()];
            const Vec2f uv1 = input.texcoords[input.faces[i].y()];
            const Vec2f uv2 = input.texcoords[input.faces[i].z()];

            for (uint32_t j = 0; j < num_micro_triangles; j++) {
                // Get barycentric coordinates of micro triangle in opacity map
                auto barycentrics = OpacityMicromap::indexToBarycentrics(j, input.subdivision_level);

                int state = 0;
                if (is_bitmap)
                    state = evaluateOpacitymapFromBitmap(input.format, std::get<std::shared_ptr<BitmapTexture>>(input.opacity_bitmap_or_function), barycentrics, uv0, uv1, uv2);
                else if (is_fbitmap)
                    state = evaluateOpacitymapFromBitmap(input.format, std::get<std::shared_ptr<FloatBitmapTexture>>(input.opacity_bitmap_or_function), barycentrics, uv0, uv1, uv2);
                else if (is_lambda)
                    state = std::get<OpacityMicromap::OpacityFunction>(input.opacity_bitmap_or_function)(barycentrics, uv0, uv1, uv2);

                int32_t index = i * num_elems_per_face + (j / num_states_per_elem);
                omm_opacity_data[index] |= state << (j % num_states_per_elem * input.format);
            }
        }

        // Reset usage counts
        if (!m_usage_counts.empty())
            m_usage_counts.clear();
        // Create usage count for building GAS
        OptixOpacityMicromapUsageCount usage_count = {
            .count = input.num_faces,
            .subdivisionLevel = input.subdivision_level,
            .format = input.format
        };
        m_usage_counts.push_back(usage_count);

        // Copy opacity buffer to device
        CUdeviceptr d_omm_opacity_data = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_omm_opacity_data), omm_opacity_data.size() * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_omm_opacity_data), omm_opacity_data.data(), omm_opacity_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));

        // Build OMM
        OptixOpacityMicromapHistogramEntry histogram = {
            .count = input.num_faces,
            .subdivisionLevel = input.subdivision_level,
            .format = input.format
        };

        // Currently, only single histogram entry is allowed
        OptixOpacityMicromapArrayBuildInput build_input = {};
        build_input.flags = build_flags;
        build_input.inputBuffer = d_omm_opacity_data;
        build_input.numMicromapHistogramEntries = 1;
        build_input.micromapHistogramEntries = &histogram;

        // Calculate memory usage for OMM
        OptixMicromapBufferSizes buffer_sizes = {};
        OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(static_cast<OptixDeviceContext>(ctx), &build_input, &buffer_sizes));

        // Setup descriptor
        std::vector<OptixOpacityMicromapDesc> omm_descs(input.num_faces);
        uint32_t offset = 0;
        for (auto& desc : omm_descs)
        {
            desc = {
                .byteOffset = offset,
                .subdivisionLevel = static_cast<uint16_t>(input.subdivision_level),
                .format = static_cast<uint16_t>(input.format)
            };
            offset += num_elems_per_face * sizeof(uint16_t);
        }

        // Copy descriptors to the device
        CUDABuffer<OptixOpacityMicromapDesc> d_omm_descs;
        d_omm_descs.copyToDevice(omm_descs);

        build_input.perMicromapDescBuffer = d_omm_descs.devicePtr();
        build_input.perMicromapDescStrideInBytes = 0;

        CUdeviceptr d_temp_buffer = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_buffers.output), buffer_sizes.outputSizeInBytes));

        m_buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
        m_buffers.temp = d_temp_buffer;
        m_buffers.tempSizeInBytes = buffer_sizes.tempSizeInBytes;

        OPTIX_CHECK(optixOpacityMicromapArrayBuild(static_cast<OptixDeviceContext>(ctx), stream, &build_input, &m_buffers));

        // Free buffers
        cuda_frees(d_omm_opacity_data, d_temp_buffer);
        d_omm_descs.free();
    }

    // ------------------------------------------------------------------
    void OpacityMicromap::build(const Context& ctx, CUstream stream, const std::vector<Input>& inputs, uint32_t build_flags)
    {
        // Clean up usage counts
        if (!m_usage_counts.empty()) 
            m_usage_counts.clear();

        // Accumurate the number of triangle faces
        size_t all_num_faces = 0;
        std::for_each(inputs.begin(), inputs.end(), [&all_num_faces](const Input& input) {
            all_num_faces += input.num_faces;
        });

        uint16_t** omm_opacity_data = new uint16_t* [all_num_faces];

        std::vector<OptixOpacityMicromapHistogramEntry> histograms;
        std::vector<OptixOpacityMicromapDesc> omm_descs(all_num_faces);

        uint32_t base_idx = 0;
        uint32_t desc_offset = 0;

        for (const auto& input : inputs)
        {
            const int32_t num_micro_triangles = 1 << (input.subdivision_level * 2);
            const size_t num_states_per_elem = 16 / input.format;

            // Check if 
            const bool is_bitmap = std::holds_alternative<std::shared_ptr<BitmapTexture>>(input.opacity_bitmap_or_function);
            const bool is_fbitmap = std::holds_alternative<std::shared_ptr<FloatBitmapTexture>>(input.opacity_bitmap_or_function);
            const bool is_lambda = std::holds_alternative<OpacityMicromap::OpacityFunction>(input.opacity_bitmap_or_function);
            ASSERT(is_bitmap || is_fbitmap || is_lambda, "Invalid format to construct opacity map");

            for (uint32_t i = base_idx; i < base_idx + input.num_faces; i++) 
            {
                omm_opacity_data[i] = new uint16_t[num_micro_triangles / num_states_per_elem];

                const Vec2f uv0 = input.texcoords[input.faces[i - base_idx].x()];
                const Vec2f uv1 = input.texcoords[input.faces[i - base_idx].y()];
                const Vec2f uv2 = input.texcoords[input.faces[i - base_idx].z()];

                for (int32_t u_tri = 0; u_tri < num_micro_triangles; u_tri++)
                {
                    auto barycentrics = OpacityMicromap::indexToBarycentrics(u_tri, input.subdivision_level);

                    int state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
                    if (is_bitmap)
                        state = evaluateOpacitymapFromBitmap(input.format, std::get<std::shared_ptr<BitmapTexture>>(input.opacity_bitmap_or_function), barycentrics, uv0, uv1, uv2);
                    else if (is_fbitmap)
                        state = evaluateOpacitymapFromBitmap(input.format, std::get<std::shared_ptr<FloatBitmapTexture>>(input.opacity_bitmap_or_function), barycentrics, uv0, uv1, uv2);
                    else
                        state = state = std::get<OpacityMicromap::OpacityFunction>(input.opacity_bitmap_or_function)(barycentrics, uv0, uv1, uv2);

                    omm_opacity_data[i][u_tri / num_states_per_elem] |= state << (u_tri % num_states_per_elem * input.format);
                }

                // Setup descriptor for all triangles
                OptixOpacityMicromapDesc omm_desc = {
                    .byteOffset = desc_offset,
                    .subdivisionLevel = static_cast<uint16_t>(input.subdivision_level),
                    .format = static_cast<uint16_t>(input.format)
                };
                desc_offset += sizeof(uint16_t) * (num_micro_triangles / num_states_per_elem);
                omm_descs[i] = omm_desc;
            }
            base_idx += input.num_faces;

            // Prepare usage counts for each input
            OptixOpacityMicromapUsageCount usage_count = {
                .count = input.num_faces, 
                .subdivisionLevel = input.subdivision_level,
                .format = input.format
            };
            m_usage_counts.emplace_back(usage_count);

            // Add histogram
            OptixOpacityMicromapHistogramEntry histogram = {
                .count = input.num_faces,
                .subdivisionLevel = input.subdivision_level, 
                .format = input.format
            };
            histograms.emplace_back(histogram);
        }

        // Copy opacity buffer to device
        CUdeviceptr d_omm_opacity_data = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_omm_opacity_data), sizeof(omm_opacity_data)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_omm_opacity_data), omm_opacity_data, sizeof(omm_opacity_data), cudaMemcpyHostToDevice));

        // Prepare build input
        OptixOpacityMicromapArrayBuildInput build_input = {
            .flags = build_flags,
            .inputBuffer = d_omm_opacity_data,
            .numMicromapHistogramEntries = static_cast<uint32_t>(histograms.size()),
            .micromapHistogramEntries = histograms.data()
        };

        // Calculate memory usage for OMM
        OptixMicromapBufferSizes buffer_sizes{};
        OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(static_cast<OptixDeviceContext>(ctx), &build_input, &buffer_sizes));
        
        // Copy descriptors to the device
        CUDABuffer<OptixOpacityMicromapDesc> d_omm_descs;
        d_omm_descs.copyToDevice(omm_descs);

        build_input.perMicromapDescBuffer = d_omm_descs.devicePtr();
        build_input.perMicromapDescStrideInBytes = 0;

        // Allocate buffers to create micromap
        CUdeviceptr d_temp_buffer = 0;
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_buffers.output), buffer_sizes.outputSizeInBytes));

        m_buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
        m_buffers.temp = d_temp_buffer;
        m_buffers.tempSizeInBytes = buffer_sizes.tempSizeInBytes;

        OPTIX_CHECK(optixOpacityMicromapArrayBuild(static_cast<OptixDeviceContext>(ctx), stream, &build_input, &m_buffers));

        // Clean up temporaly buffers
        cuda_frees(d_omm_opacity_data, d_temp_buffer);
        d_omm_descs.free();
    }

    // ------------------------------------------------------------------
    OptixBuildInputOpacityMicromap OpacityMicromap::getBuildInputForGAS() const
    {
        // Check if OMM has already been builded
        ASSERT(m_buffers.output != 0, "OMM has not been builded yet.");

        // Accumurate all usage count
        uint32_t all_count = 0;
        std::for_each(m_usage_counts.begin(), m_usage_counts.end(), [&all_count](const OptixOpacityMicromapUsageCount& usage_count)
            {
                all_count += usage_count.count;
            });

        // Prepare OMM indices 
        std::vector<uint16_t> omm_indices(all_count);
        std::iota(omm_indices.begin(), omm_indices.end(), 0);

        CUDABuffer<uint16_t> d_omm_indices;
        d_omm_indices.copyToDevice(omm_indices);

        OptixBuildInputOpacityMicromap omm_input = {};
        omm_input.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
        omm_input.opacityMicromapArray = m_buffers.output;
        omm_input.indexBuffer = d_omm_indices.devicePtr();
        omm_input.indexSizeInBytes = sizeof(uint16_t);
        omm_input.numMicromapUsageCounts = static_cast<uint32_t>(m_usage_counts.size());
        omm_input.micromapUsageCounts = m_usage_counts.data();

        return omm_input;
    }

    // ------------------------------------------------------------------
    template <typename U>
    int OpacityMicromap::evaluateOpacitymapFromBitmap(
        OptixOpacityMicromapFormat format,
        const std::shared_ptr<U>& bitmap,
        const OpacityMicromap::MicroBarycentrics& bc,
        const Vec2f& uv0,
        const Vec2f& uv1,
        const Vec2f& uv2
    )
    {
        using Pixel = std::conditional_t<std::is_same_v<U, BitmapTexture>, Vec4u, Vec4f>;

        const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, bc.uv0);
        const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, bc.uv1);
        const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, bc.uv2);

        int32_t w = bitmap->width();
        int32_t h = bitmap->height();

        /// TODO: Rasterize micro-triangle to evaluate transparency
        Vec2i p0 = Vec2i{ static_cast<int>(clamp(bary0.x(), 0.0f, 0.999f) * w), static_cast<int>(clamp(bary0.y(), 0.0f, 0.999f) * h) };
        Vec2i p1 = Vec2i{ static_cast<int>(clamp(bary1.x(), 0.0f, 0.999f) * w), static_cast<int>(clamp(bary1.y(), 0.0f, 0.999f) * h) };
        Vec2i p2 = Vec2i{ static_cast<int>(clamp(bary2.x(), 0.0f, 0.999f) * w), static_cast<int>(clamp(bary2.y(), 0.0f, 0.999f) * h) };

        Vec2i corner_min = Vec2i{ min(min(p0.x(), p1.x()), p2.x()), min(min(p0.y(), p1.y()), p2.y())};
        Vec2i corner_max = Vec2i{ max(max(p0.x(), p1.x()), p2.x()), max(max(p0.y(), p1.y()), p2.y())};

        auto calcArea = [](const Vec2i& a, const Vec2i& b, const Vec2i& c) -> int {
            return cross(b - a, c - a);
        };

        // Accmulate transparency inside micro-triangle by scanning all pixels inside bounding box that just covers the triangle
        int32_t num_pixels_in_triangle = 0;
        int32_t num_transparent_pixels = 0;
        for (int32_t x = corner_min.x(); x <= corner_max.x(); x++) {
            for (int32_t y = corner_min.y(); y <= corner_max.y(); y++) {
                Vec2i p(x, y);
                // The pixel on/inside the micro triangle
                auto area01 = calcArea(p, p0, p1);
                auto area12 = calcArea(p, p1, p2);
                auto area20 = calcArea(p, p2, p0);
                // Accumerate the number of transparent pixels inside the micro triangle
                if ((area01 >= 0 && area12 >= 0 && area20 >= 0) || (area01 <= 0 && area12 <= 0 && area20 <= 0)) {
                    Pixel color = bitmap->eval(p);
                    num_pixels_in_triangle++;
                    if (color.w() == 0)
                        num_transparent_pixels++;
                }
            }
        }

        if (num_transparent_pixels == num_pixels_in_triangle)
            return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
        else if (num_transparent_pixels == 0)
            return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
        else
        {
            if (format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
            else // Treat micro triangle as opaque when the state is controlled with 1 bit (0 or 1)
                return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
        }
    }

    // ------------------------------------------------------------------
    OpacityMicromap::MicroBarycentrics OpacityMicromap::indexToBarycentrics(uint32_t index, uint32_t subdivision_level)
    {
        float2 bary0, bary1, bary2;
        optixMicromapIndexToBaseBarycentrics(index, subdivision_level, bary0, bary1, bary2);
        return MicroBarycentrics{ bary0, bary1, bary2 };
    }
} // namespace prayground