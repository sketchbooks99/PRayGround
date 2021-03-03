#pragma once

#include <optix.h>
#include "core_util.h"

namespace pt {

class Pipeline {
public:
    explicit Pipeline() {
        m_options.usesMotionBlur = false;
        m_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_ALLOW_SINGLE_GAS;
        m_options.numPayloadValues = 5;
        m_options.numAttributeValues = 2;
    }
    explicit Pipeline(OptixPipelineCompileOptions op) : m_options(op) {}

    void set_options(const OptixPipelineCompileOptions& op) {
        m_options = op;
    }
    void create(const OptixDeviceContext &ctx, const std::vector<OptixProgramGroup>& prg_groups) {

    }

private:
    OptixPipelineCompileOptions m_options;
    OptixPipeline m_pipeline;
};

}