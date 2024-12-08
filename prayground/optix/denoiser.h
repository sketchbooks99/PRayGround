#pragma once 

#include <optix.h>
#include <vector>
#include <prayground/optix/context.h>
#include <prayground/core/bitmap.h>

namespace prayground {

class Denoiser {
public:
    struct Data {
        uint32_t width;
        uint32_t height;
        float* color = nullptr;
        float* albedo = nullptr;
        float* normal = nullptr;
        float* flow = nullptr;
        std::vector<float*> aovs;
        std::vector<float*> outputs;
    };

    Denoiser();

    void init(const Context& ctx,
              const Data& data, 
              uint32_t tile_width = 0, 
              uint32_t tile_height = 0, 
              bool kp_mode = false,
              bool is_temporal = false);
    void run();
    void update(const Data& data);
    void draw(const Data& data);
    void draw(const Data& data, int x, int y);
    void draw(const Data& data, int x, int y, int w, int h);
    void write(const Data& data, const std::filesystem::path& filepath);
    // Copy results from GPU to host memory
    void copyFromDevice();
    // Finialize denoiser
    void destroy();

    // test flow vectors: flow is applied to noisy input image and written back to result
    // no denoising
    void copyFlowFromDevice();
private:
    OptixDenoiser       m_denoiser { nullptr };
    OptixDenoiserParams m_params { };

    bool                is_temporal { false };

    CUdeviceptr         m_intensity     { 0 };
    CUdeviceptr         m_avg_color     { 0 };
    CUdeviceptr         m_scratch       { 0 };
    uint32_t            m_scratch_size  { 0 };
    CUdeviceptr         m_state         { 0 };
    uint32_t            m_state_size    { 0 };

    uint32_t            m_tile_width    { 0 };
    uint32_t            m_tile_height   { 0 };
    uint32_t            m_overlap       { 0 };

#if OPTIX_VERSION <= 70200
    OptixImage2D        m_inputs[3] {};
    OptixImage2D        m_output    {};
    float*              m_host_output = nullptr;
#else
    OptixDenoiserGuideLayer           m_guide_layer { };
    std::vector< OptixDenoiserLayer > m_layers;
#endif
    std::vector< float* >             m_host_outputs;

    // For drawing result 
    FloatBitmap m_viewer;
};

}