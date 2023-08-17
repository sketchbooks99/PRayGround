#include "denoiser.h"
#include <prayground/app/app_runner.h>
#include <prayground/core/util.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/util.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>

namespace prayground {

/// @todo Support Optix ~7.2

// --------------------------------------------------------------------
static inline float catmullRom(float p[4], float t)
{
    return p[1] * 0.5f * t * (p[2] - p[0] + t * (2.0f * p[0] - 5.0f * p[1] + 4.0f * p[2] - p[3] + t * (3.0f * (p[1] - p[2]) + p[3] - p[0])));
}

// --------------------------------------------------------------------
static void addFlow(
    float4* result, 
    const float4* image, const float4* flow, 
    uint32_t width, uint32_t height, 
    uint32_t x, uint32_t y)
{
    float dst_x = (float)x - flow[x + y * width].x;
    float dst_y = (float)y - flow[x + y * width].y;

    float x0 = dst_x - 1.0f;
    float y0 = dst_y - 1.0f;

    float r[4][4], g[4][4], b[4][4];
    for (int j = 0; j < 4; j++)
    {
        for (int k = 0; k < 4; k++)
        {
            int tx = static_cast<int>(x0) + k;
            if (tx < 0)
                tx = 0;
            else if (tx >= (int)width )
                tx = width - 1;

            int ty = static_cast<int>(y0) + j;
            if (ty < 0)
                ty = 0;
            else if (ty >= (int)height)
                ty = height - 1;
            
            r[j][k] = image[tx + ty * width].x;
            g[j][k] = image[tx + ty * width].y;
            b[j][k] = image[tx + ty * width].z;
        }
    }
    float tx = dst_x <= 0.0f ? 0.0f : dst_x - floorf(dst_x);

    r[0][0] = catmullRom(r[0], tx);
    r[0][1] = catmullRom(r[1], tx);
    r[0][2] = catmullRom(r[2], tx);
    r[0][3] = catmullRom(r[3], tx);

    g[0][0] = catmullRom( g[0], tx );
    g[0][1] = catmullRom( g[1], tx );
    g[0][2] = catmullRom( g[2], tx );
    g[0][3] = catmullRom( g[3], tx );

    b[0][0] = catmullRom( b[0], tx );
    b[0][1] = catmullRom( b[1], tx );
    b[0][2] = catmullRom( b[2], tx );
    b[0][3] = catmullRom( b[3], tx );

    float ty = dst_y <= 0.0f ? 0.0f : dst_y - floorf(dst_y);

    result[y * width + x].x = catmullRom(r[0], ty);
    result[y * width + x].y = catmullRom(g[0], ty);
    result[y * width + x].z = catmullRom(b[0], ty);
}

// --------------------------------------------------------------------
static OptixImage2D createOptixImage2D(const int width, const int height, const float* hmem = nullptr)
{
    OptixImage2D oi;

    const uint64_t frame_byte_size = width * height * sizeof(float4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&oi.data), frame_byte_size));
    if (hmem)
    {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(oi.data), hmem, frame_byte_size, cudaMemcpyHostToDevice));
    }
    oi.width = width; 
    oi.height = height;
    oi.rowStrideInBytes = width * sizeof(float4);
    oi.pixelStrideInBytes = sizeof(float4);
    oi.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return oi;
}

// --------------------------------------------------------------------
Denoiser::Denoiser()
{
    std::string version = pgGetOptixVersionString(OPTIX_VERSION);
    if (OPTIX_VERSION <= 70200)
    {
        pgLogFatal("Currently, Denoiser is supported with OptiX 7.3~ (Your version is " + version + ".");
        std::exit(0);
    }
}

// --------------------------------------------------------------------
void Denoiser::init(
    const Context& ctx,
    const Data& data,
    uint32_t tile_width, uint32_t tile_height,
    bool kp_mode, bool is_temporal)
{
#if OPTIX_VERSION <= 70200
    UNIMPLEMENTED();
#else
    ASSERT(data.color, "data.color must be set.");
    ASSERT(data.outputs.size() >= 1, "data.outputs must have at least one buffer.");
    ASSERT(data.width, "data.width must be greater than 0.");
    ASSERT(data.height, "data.height must be greater than 0.");
    ASSERT(!data.normal || data.albedo, "Currently albedo is required if normal input is given");
    ASSERT((tile_width == 0 && tile_height == 0) || (tile_width > 0 && tile_height > 0), "tile size must be > 0 for width and height.");

    m_host_outputs = data.outputs;
    this->is_temporal = is_temporal;

    m_tile_width = tile_width > 0 ? tile_width : data.width;
    m_tile_height = tile_height > 0 ? tile_height : data.height;

    // Create denoiser
    {
        OptixDenoiserOptions options = {};
        options.guideAlbedo = data.albedo ? 1 : 0;
        options.guideNormal = data.normal ? 1 : 0;

        OptixDenoiserModelKind model_kind;
        if (kp_mode || data.aovs.size() > 0)
        {
            ASSERT(!is_temporal, "temporal mode must be enabled.");
            model_kind = OPTIX_DENOISER_MODEL_KIND_AOV;
        }
        else
        {
            model_kind = is_temporal ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
        }
        OPTIX_CHECK(optixDenoiserCreate(static_cast<OptixDeviceContext>(ctx), model_kind, &options, &m_denoiser));
    }

    {
        OptixDenoiserSizes denoiser_sizes;

        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            m_denoiser,
            m_tile_width,
            m_tile_height,
            &denoiser_sizes));

        if (tile_width == 0)
        {
            m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
            m_overlap = 0;
        }
        else
        {
            m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withOverlapScratchSizeInBytes);
            m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
        }

        if (data.aovs.size() == 0 && kp_mode == false)
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&m_intensity),
                sizeof(float)
            ));
        }
        else
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&m_avg_color),
                3 * sizeof(float)
            ));
        }

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_scratch),
            m_scratch_size
        ));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_state),
            denoiser_sizes.stateSizeInBytes
        ));

        m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

        OptixDenoiserLayer layer = {};
        layer.input = createOptixImage2D(data.width, data.height, data.color);
        layer.output = createOptixImage2D(data.width, data.height);
        if (is_temporal)
        {
            void* flowmem;
            CUDA_CHECK(cudaMalloc(&flowmem, data.width * data.height * sizeof(float4)));
            CUDA_CHECK(cudaMemset(flowmem, 0, data.width * data.height * sizeof(float4)));
            m_guide_layer.flow = { (CUdeviceptr)flowmem, data.width, data.height, (uint32_t)(data.width * sizeof(float4)), (uint32_t)sizeof(float4), OPTIX_PIXEL_FORMAT_FLOAT4 };

            layer.previousOutput = layer.input;
        }
        m_layers.push_back(layer);

        if (data.albedo)
            m_guide_layer.albedo = createOptixImage2D(data.width, data.height, data.albedo);
        if (data.normal)
            m_guide_layer.normal = createOptixImage2D(data.width, data.height, data.normal);

        for (size_t i = 0; i < data.aovs.size(); i++)
        {
            layer.input = createOptixImage2D(data.width, data.height, data.aovs[i]);
            layer.output = createOptixImage2D(data.width, data.height);
            if (is_temporal)
                layer.previousOutput = layer.input;
            m_layers.push_back(layer);
        }
    }

    // Setup denoiser
    {
        OPTIX_CHECK(optixDenoiserSetup(
            m_denoiser,
            nullptr,
            m_tile_width + 2 * m_overlap,
            m_tile_height + 2 * m_overlap,
            m_state,
            m_state_size,
            m_scratch,
            m_scratch_size
        ));

#if OPTIX_VERSION <= 70400
        m_params.denoiseAlpha = 0;
#elif OPTIX_VERSION < 80000 
        m_params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
#endif
        m_params.hdrIntensity = m_intensity;
        m_params.hdrAverageColor = m_avg_color;
        m_params.blendFactor = 0.0f;
    }

    m_viewer.allocate(PixelFormat::RGBA, (int)data.width, (int)data.height);
#endif
}

// --------------------------------------------------------------------
void Denoiser::run()
{
#if OPTIX_VERSION <= 70200
    UNIMPLEMENTED();
#else 
    if (m_intensity)
    {
        OPTIX_CHECK(optixDenoiserComputeIntensity(
            m_denoiser, 
            nullptr, 
            &m_layers[0].input, 
            m_intensity, 
            m_scratch, 
            m_scratch_size
        ));
    }

    if (m_avg_color)
    {
        OPTIX_CHECK(optixDenoiserComputeAverageColor(
            m_denoiser, 
            nullptr, 
            &m_layers[0].input, 
            m_avg_color, 
            m_scratch,
            m_scratch_size
        ));
    }

    OPTIX_CHECK(optixUtilDenoiserInvokeTiled(
        m_denoiser, 
        nullptr, 
        &m_params, 
        m_state, 
        m_state_size, 
        &m_guide_layer, 
        m_layers.data(), 
        static_cast<unsigned int>(m_layers.size()), 
        m_scratch, 
        m_scratch_size, 
        m_overlap, 
        m_tile_width, 
        m_tile_height
    ));

    CUDA_SYNC_CHECK();
#endif
}

// --------------------------------------------------------------------
void Denoiser::update(const Data& data)
{
#if OPTIX_VERSION <= 70200
    UNIMPLEMENTED();
#else
    ASSERT(data.color, "data.color is nullptr");
    ASSERT(data.outputs.size() >= 1, "The number of outputs must be greater than 1.");
    ASSERT(data.width > 0 && data.height, "The dimensions of denoiser must be greater than 1x1.");
    ASSERT(!data.normal || data.albedo, "Currently albedo is required if normal input is given");

    m_host_outputs = data.outputs;

    CUDA_CHECK(cudaMemcpy((void*)m_layers[0].input.data, data.color, data.width * data.height * sizeof(float4), cudaMemcpyHostToDevice));

    if (is_temporal)
    {
        CUDA_CHECK(cudaMemcpy((void*)m_guide_layer.flow.data, data.flow, data.width * data.height * sizeof(float4), cudaMemcpyHostToDevice));
        m_layers[0].previousOutput = m_layers[0].output;
    }

    if (data.albedo)
        CUDA_CHECK(cudaMemcpy((void*)m_guide_layer.albedo.data, data.albedo, data.width * data.height * sizeof(float4), cudaMemcpyHostToDevice));
    
    if (data.normal)
        CUDA_CHECK(cudaMemcpy((void*)m_guide_layer.normal.data, data.normal, data.width * data.height * sizeof(float4), cudaMemcpyHostToDevice));

    for (size_t i = 0; i < data.aovs.size(); i++)
    {
        CUDA_CHECK(cudaMemcpy((void*)m_layers[i].input.data, data.aovs[i], data.width * data.height * sizeof(float4), cudaMemcpyHostToDevice));
        if (is_temporal)
            m_layers[i].previousOutput = m_layers[i].output;
    }
#endif
}

// --------------------------------------------------------------------
void Denoiser::draw(const Data& data)
{
    draw(data, 0, 0, data.width, data.height);
}

void Denoiser::draw(const Data& data, int x, int y)
{
    draw(data, x, y, data.width, data.height);
}

void Denoiser::draw(const Data& data, int x, int y, int w, int h)
{
    m_viewer.setData(data.outputs[0], 0, 0, data.width, data.height);
    m_viewer.draw(x, y, w, h);
}

// --------------------------------------------------------------------
void Denoiser::destroy()
{
#if OPTIX_VERSION <= 70200
    UNIMPLEMENTED();
#else
    OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_intensity)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_avg_color)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_scratch)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_guide_layer.albedo.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_guide_layer.normal.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_guide_layer.flow.data)));
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_layers[i].input.data)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_layers[i].output.data)));
    }
#endif
}

// --------------------------------------------------------------------
void Denoiser::copyFlowFromDevice()
{
#if OPTIX_VERSION <= 70200
    UNIMPLEMENTED();
#else
    if (m_layers.size() == 0)
        return;
    
    const uint64_t frame_byte_size = m_layers[0].output.width * m_layers[0].output.height * sizeof(float4);

    const float4* device_flow = (float4*)m_guide_layer.flow.data;
    if (!device_flow)
        return;
    float4* flow = new float4[frame_byte_size];
    CUDA_CHECK(cudaMemcpy(flow, device_flow, frame_byte_size, cudaMemcpyDeviceToHost));

    float4* image = new float4[frame_byte_size];

    for (size_t i = 0; i < m_layers.size(); i++)
    {
        CUDA_CHECK(cudaMemcpy(image, (float4*)m_layers[i].input.data, frame_byte_size, cudaMemcpyDeviceToHost));

        for (uint32_t y = 0; y < m_layers[i].output.height; y++)
            for (uint32_t x = 0; x < m_layers[i].output.width; x++)
                addFlow((float4*)m_host_outputs[i], image, flow, m_layers[i].input.width, m_layers[i].input.height, x, y);
    }

    delete[] image;
    delete[] flow;
#endif
}

// --------------------------------------------------------------------
void Denoiser::copyFromDevice()
{
#if OPTIX_VERSION <= 70200
    UNIMPLEMENTED();
#else
    const uint64_t frame_byte_size = m_layers[0].output.width * m_layers[0].output.height * sizeof(float4);
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        CUDA_CHECK(cudaMemcpy(
            m_host_outputs[i], 
            reinterpret_cast<void*>(m_layers[i].output.data), 
            frame_byte_size, 
            cudaMemcpyDeviceToHost));
    }
#endif
}

} // ::prayground