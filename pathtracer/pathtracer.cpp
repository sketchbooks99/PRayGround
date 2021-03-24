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
#include <include/optix/module.h>
#include <include/optix/pipeline.h>
#include <include/optix/sbt.h>
#include <include/optix/program.h>
#include <include/optix/macros.h>

// include application utilities
#include <include/core/util.h>
#include <include/core/cudabuffer.h>
#include <include/core/pathtracer.h>
#include <include/core/scene.h>
#include <include/core/primitive.h>

// Header file describe the scene
#include "scene_file.h"

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state 
int32_t mouse_button = -1;
int32_t samples_per_launch = 4;

// ========== GLFW callbacks ==========
static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if(action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }  
} 

static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    pt::Params* params = static_cast<pt::Params*>( glfwGetWindowUserPointer( window ));

    if(mouse_button == GLFW_MOUSE_BUTTON_LEFT ) {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height);
        camera_changed = true;
    }
    else if(mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
}

static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    if( minimized )
        return;
    
    sutil::ensureMinimumSize( res_x, res_y );

    pt::Params* params = static_cast<pt::Params*>( glfwGetWindowUserPointer( window ));
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
}

static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}

static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}

static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll)
{
    if( trackball.wheelEvent( (int)yscroll ))
        camera_changed = true;
}

// ========== Helper functions ==========

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>       File for image output\n";
    std::cerr << "         --launch-samples | -s        Numper of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop              Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>       Set image dimensions; defautlt to 768x768\n";
    std::cerr << "         --help | -h                  Print this usage message\n";
    exit( 0 );
}

void handleCameraUpdate( pt::Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, pt::Params& params)
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &params.accum_buffer ),
        params.width * params.height * sizeof( float4 )
    ));
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, pt::Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4> & output_buffer,   // output buffer
                     pt::Params& params,                                // Launch parameters of OptiX
                     const CUdeviceptr& d_params,                       // Device side pointer of launch params
                     const CUstream& stream,                            // CUDA stream
                     const OptixPipeline& pipeline,                     // Pipeline of OptiX
                     const OptixShaderBindingTable& sbt                 // Shader binding table
                     )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast<void*>( d_params ),
        &params, sizeof( pt::Params ),
        cudaMemcpyHostToDevice, stream
    ));

    // Start ray tracing on device.
    OPTIX_CHECK( optixLaunch(
        pipeline, 
        stream,
        d_params,
        sizeof( pt::Params ),
        &sbt,
        params.width,
        params.height,
        5
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;
    int framebuf_res_y = 0;
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    /// Load scene from \c scene_file.h 
    pt::Scene scene = my_scene(); 

    pt::Params params;
    params.width = scene.width();
    params.height = scene.height();
    params.samples_per_launch = scene.samples_per_launch();
    params.max_depth = scene.depth();   
    // =======================================================================
    // ↓ SAMPLE CODE 
    // =======================================================================
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
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            scene.set_width(w);
            scene.set_height(h);
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try 
    {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));

        // Create device context.
        OptixDeviceContext optix_context;
        CUcontext cu_ctx = 0;
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optix_context));

        pt::Message("main(): Created device context");

        // Build children instance ASs that contain the geometry AS. 
        std::vector<OptixInstance> instances;
        // std::vector<pt::AccelData> accels;
        unsigned int sbt_base_offset = 0;       
        unsigned int instance_id = 0;
        for (auto &ps : scene.primitive_instances()) {
            // accels.push_back(pt::AccelData());
            pt::AccelData accel = {};
            pt::build_gas(optix_context, accel, ps);
            /// New OptixInstance are pushed back to \c instances
            pt::build_ias(optix_context, accel, ps, sbt_base_offset, instance_id, instances);
        }

        std::cout << "main(): Builded children instance ASs that contain the geometry AS" << std::endl;

        // Create all instances on the device.
        pt::CUDABuffer<OptixInstance> d_instances;
        d_instances.alloc_copy(instances);

        // Prepare build input for instances.
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances = d_instances.d_ptr();
        instance_input.instanceArray.numInstances = (unsigned int) instances.size();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE; 
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, 
            &accel_options, 
            &instance_input, 
            1, 
            &ias_buffer_sizes ));

        // Allocate buffer to build acceleration structure
        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_temp_buffer), 
            ias_buffer_sizes.tempSizeInBytes ));
        CUdeviceptr d_ias_output_buffer; 
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_ias_output_buffer), 
            ias_buffer_sizes.outputSizeInBytes ));
        
        OptixTraversableHandle ias_handle = 0;
        // Build instance AS contains all GASs to describe the scene.
        OPTIX_CHECK(optixAccelBuild(
            optix_context, 
            0,                  // CUDA stream
            &accel_options, 
            &instance_input, 
            1,                  // num build inputs
            d_temp_buffer, 
            ias_buffer_sizes.tempSizeInBytes, 
            d_ias_output_buffer, 
            ias_buffer_sizes.outputSizeInBytes, 
            &ias_handle, 
            nullptr,            // emitted property list
            0                   // num emitted properties
        ));

        pt::cuda_free(d_temp_buffer);
        d_instances.free();

        // Prepare the pipeline
        std::string params_name = "params";
        pt::Pipeline pt_pipeline(params_name);
        // Create module 
        pt::Module pt_module("optix/pathtracer.cu");
        pt_module.create(optix_context, pt_pipeline.compile_options());

        // Create program groups
        OptixShaderBindingTable optix_sbt = {};
        std::vector<OptixProgramGroup> program_groups;

        // Raygen programs
        pt::ProgramGroup raygen_program(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
        raygen_program.create(optix_context, pt::ProgramEntry((OptixModule)pt_module, RG_FUNC_STR("raygen")));
        program_groups.emplace_back(raygen_program);
        // Create sbt for raygen program
        pt::CUDABuffer<pt::RayGenRecord> d_raygen_record;
        pt::RayGenRecord rg_sbt = {};
        OPTIX_CHECK(optixSbtRecordPackHeader((OptixProgramGroup)raygen_program, &rg_sbt));
        d_raygen_record.alloc_copy(&rg_sbt, sizeof(pt::RayGenRecord));
        pt::Message("main() : Created raygen programs");

        // Miss programs
        std::vector<pt::ProgramGroup> miss_programs(RAY_TYPE_COUNT, pt::ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_MISS));
        miss_programs[0].create(optix_context, pt::ProgramEntry((OptixModule)pt_module, MS_FUNC_STR("radiance")));      // miss radiance
        miss_programs[1].create(optix_context, pt::ProgramEntry(nullptr, nullptr));                                     // miss occlusion
        std::copy(miss_programs.begin(), miss_programs.end(), std::back_inserter(program_groups)); 

        // Create sbt for miss programs
        pt::CUDABuffer<pt::MissRecord> d_miss_record;
        pt::MissRecord ms_sbt[RAY_TYPE_COUNT];
        for (int i=0; i<RAY_TYPE_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader((OptixProgramGroup)miss_programs[i], &ms_sbt[i]));
            ms_sbt[i].data.bg_color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        }
        d_miss_record.alloc_copy(ms_sbt, sizeof(pt::MissRecord)*RAY_TYPE_COUNT);
        pt::Message("main() : Created miss programs");

        // Create the sbt for raygen and miss programs
        optix_sbt.raygenRecord = d_raygen_record.d_ptr();
        optix_sbt.missRecordBase = d_miss_record.d_ptr();
        optix_sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(pt::MissRecord));
        optix_sbt.missRecordCount = RAY_TYPE_COUNT;
        
        // HitGroup programs
        scene.create_hitgroup_programs(optix_context, (OptixModule)pt_module);
        std::vector<pt::ProgramGroup> hitgroup_program = scene.hitgroup_programs();
        std::copy(hitgroup_program.begin(), hitgroup_program.end(), std::back_inserter(program_groups));

        // Create pipeline
        pt_pipeline.create(optix_context, program_groups);

        // Create sbt for all hitgroup programs
        scene.create_hitgroup_sbt((OptixModule)pt_module, optix_sbt);

        // Initialize launch parameters of OptiX
        CUdeviceptr d_params;
        CUstream stream;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.accum_buffer),
                scene.width() * scene.height() * sizeof(float4)));
        params.frame_buffer = nullptr;
        params.subframe_index = 0u;
        params.handle = ias_handle;
        pt::Message("ias_handle", ias_handle);
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(pt::Params)));

        if (outfile.empty())
        {
            pt::Message("Starting path tracer...");
            GLFWwindow* window = sutil::initUI("Path Tracer", params.width, params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &params);

            // Render loop
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    params.width, 
                    params.height
                );

                output_buffer.setStream(stream);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do 
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, params, d_params, stream, 
                                   (OptixPipeline)pt_pipeline, optix_sbt);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    params.subframe_index++;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }
            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                params.width,
                params.height
            );

            handleCameraUpdate(params);
            handleResize(output_buffer, params);
            launchSubframe(output_buffer, params, d_params, stream, 
                                   (OptixPipeline)pt_pipeline, optix_sbt);

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage(outfile.c_str(), buffer, false);

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        // Crean up
        pt_pipeline.destroy();
        for (auto &pg : program_groups) OPTIX_CHECK(optixProgramGroupDestroy(pg));
        pt_module.destroy();
        OPTIX_CHECK(optixDeviceContextDestroy(optix_context));

        pt::cuda_frees(optix_sbt.raygenRecord,
                   optix_sbt.missRecordBase,
                   optix_sbt.hitgroupRecordBase,
                   params.accum_buffer, 
                   d_params);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caugh exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}