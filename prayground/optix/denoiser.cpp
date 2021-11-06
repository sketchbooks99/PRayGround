#include "denoiser.h"
#include <prayground/core/util.h>
#include <prayground/optix/macros.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>

namespace prayground {

// --------------------------------------------------------------------
Denoiser::Denoiser()
{

}

// --------------------------------------------------------------------
void Denoiser::create(
    const Context& ctx,
    const Data& data, 
    uint32_t tile_width, uint32_t tile_height, 
    bool kp_mode, bool is_temporal)
{
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
            reinterpret_cast<void**>(&m_scratch), 
            denoiser_sizes.stateSizeInBytes
        ));

        m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

        OptixDenoiserLayer layer = {};
    }
}

// --------------------------------------------------------------------
void Denoiser::run()
{

}

// --------------------------------------------------------------------
void Denoiser::update(const Data& data)
{

}

// --------------------------------------------------------------------
void Denoiser::destroy()
{

}

}