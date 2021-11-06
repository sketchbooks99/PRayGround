#pragma once 

#include <optix.h>
#include <vector>
#include <prayground/optix/context.h>

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

    void create(const Context& ctx,
                const Data& data, 
                uint32_t tile_width = 0, 
                uint32_t tile_height = 0, 
                bool kp_mode = false,
                bool is_temporal = false);
    void run();
    void update(const Data& data);
    void destroy();
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

    OptixDenoiserGuideLayer           m_guide_layer { };
    std::vector< OptixDenoiserLayer > m_layers;
    std::vector< float* >             m_host_outputs;
};

}