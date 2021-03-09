#include <glad/glad.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

// include optix utilities
#include "core/cudabuffer.h"
#include "core/program.h"
#include "core/module.h"
#include "core/pipeline.h"
#include "core/program.h"

// include application utilities
#include "core/core_util.h"
#include "core/ray.h"
#include "core/shape.h"
#include "core/pathtracer.h"
#include "core/material.h"

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state 
int32_t mouse_button = -1;
int32_t samples_per_launch = 4;

int main(int argc, char* argv[]) {
    OptixDeviceContext ctx;

    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    std::string outfile;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--file" || "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[i++];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int w, h;
            sutil::parseDimensions(dims_arg.c_str, w, h);
            state.params.width = w;
            state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
    }
}