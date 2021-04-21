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
#include "optix/module.h"
#include "optix/pipeline.h"
#include "optix/sbt.h"
#include "optix/program.h"
#include "optix/macros.h"

// include application utilities
#include "core/util.h"
#include "core/cudabuffer.h"
#include "core/optix-raytracer.h"
#include "core/scene.h"
#include "core/primitive.h"

// Header file describe the scene
#include "scene_config.h"

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
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
    oprt::Params* params = static_cast<oprt::Params*>( glfwGetWindowUserPointer( window ));

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

    oprt::Params* params = static_cast<oprt::Params*>( glfwGetWindowUserPointer( window ));
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

void initLaunchParams( 
    const OptixTraversableHandle& gas_handle,
    CUstream& stream,
    oprt::Params& params,
    CUdeviceptr& d_params
)
{
    CUDA_CHECK (cudaMalloc(
        reinterpret_cast<void**>( &params.accum_buffer ),
        params.width * params.height * sizeof( float4 )
    ));
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.subframe_index = 0u;

    params.handle         = gas_handle;

    CUDA_CHECK( cudaStreamCreate( &stream) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( oprt::Params ) ) );
}

void handleCameraUpdate( oprt::Params& params, sutil::Camera& camera )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, oprt::Params& params)
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

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, oprt::Params& params, sutil::Camera& camera )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params, camera );
    handleResize( output_buffer, params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer,   // output buffer
                     oprt::Params& params,                                // Launch parameters of OptiX
                     const CUdeviceptr& d_params,                       // Device side pointer of launch params
                     const CUstream& stream,                            // CUDA stream
                     const oprt::Pipeline& pipeline,                     // Pipeline of OptiX
                     OptixShaderBindingTable& sbt                 // Shader binding table 
) {
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast<void*>( d_params ),
        &params, sizeof( oprt::Params ),
        cudaMemcpyHostToDevice, stream
    ));

    // Start ray tracing on device.
    OPTIX_CHECK( optixLaunch(
        (OptixPipeline)pipeline, 
        stream,
        d_params,
        sizeof( oprt::Params ),
        &sbt,
        params.width,
        params.height,
        1
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

void initCameraState(sutil::Camera& camera)
{
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    oprt::Params params = {};
    CUdeviceptr d_params;

    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    CUstream stream = 0;

    OptixShaderBindingTable sbt = {};

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            params.width  = w;
            params.height = h;
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {

        // Initialize cuda 
        CUDA_CHECK(cudaFree(0));

        // Create device context.
        OptixDeviceContext optix_context;
        CUcontext cu_context = 0;
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cu_context, &options, &optix_context));

        // Load the scene
        oprt::Scene scene = my_scene();

        params.width                             = scene.width();
        params.height                            = scene.height();
        params.samples_per_launch                = scene.samples_per_launch();
        params.max_depth                         = scene.depth();

        // Initialize camera state.
        auto camera = scene.camera();
        initCameraState(camera);

        // Create instances that manage GAS, sbtOffset, and transform of geometries.
        std::vector<OptixInstance> instances;
        unsigned int sbt_base_offset = 0; 
        unsigned int instance_id = 0;
        std::vector<oprt::AccelData*> accels;
        for ( auto ps : scene.primitive_instances() ) {
            accels.push_back( new oprt::AccelData() );
            oprt::build_gas( optix_context, *accels.back(), ps );
            oprt::build_instances( optix_context, *accels.back(), ps, sbt_base_offset, instance_id, instances ); 
        }

        oprt::CUDABuffer<OptixInstance> d_instances;
        d_instances.alloc_copy(instances);

        // Prepare build input for instances.
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances = d_instances.d_ptr();
        instance_input.instanceArray.numInstances = (unsigned int)instances.size();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE; 
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, 
            &accel_options, 
            &instance_input, 
            1,  // num build inputs
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

        oprt::cuda_free(d_temp_buffer);
        d_instances.free();

        // createModule(optix_context, pipeline_compile_options, ptx_module);
        // Prepare the pipeline
        std::string params_name = "params";
        oprt::Pipeline pipeline(params_name);
        pipeline.set_dc_depth(2);   // The maximum call depth of direct callable programs.
        pipeline.set_cc_depth(2);   // The maximum call depth of continuation callable programs.
        pipeline.set_num_payloads(5);
        pipeline.set_num_attributes(5);
        // Create module
        oprt::Module module("optix/cuda/optix-raytracer.cu");
        module.create(optix_context, pipeline.compile_options());

        /**
         * \brief Create programs
         */
        std::vector<OptixProgramGroup> program_groups;
        // Raygen program
        auto raygen_program = oprt::createRayGenProgram( optix_context, (OptixModule)module, RG_FUNC_STR("raygen") );
        program_groups.push_back( (OptixProgramGroup)raygen_program );
        // Create and bind sbt to raygen program
        oprt::CUDABuffer<oprt::RayGenRecord> d_raygen_record;
        oprt::RayGenRecord rg_record = {};
        raygen_program.bind_record( &rg_record );
        d_raygen_record.alloc_copy( &rg_record, sizeof( oprt::RayGenRecord ) );
        
        // Miss program
        std::vector<oprt::ProgramGroup> miss_programs( RAY_TYPE_COUNT );
        miss_programs[0] = oprt::createMissProgram( optix_context, (OptixModule)module, MS_FUNC_STR("radiance") );
        miss_programs[1] = oprt::createMissProgram( optix_context, nullptr, nullptr );
        std::copy( miss_programs.begin(), miss_programs.end(), std::back_inserter( program_groups ) );
        // Create sbt for miss programs
        oprt::CUDABuffer<oprt::MissRecord> d_miss_record;
        oprt::MissRecord ms_records[RAY_TYPE_COUNT];
        for (int i=0; i<RAY_TYPE_COUNT; i++) {
            miss_programs[i].bind_record( &ms_records[i] );
            ms_records[i].data.bg_color = scene.bgcolor();  
        }
        d_miss_record.alloc_copy( ms_records, sizeof(oprt::MissRecord) * RAY_TYPE_COUNT );

        // Bind sbts to raygen and miss program
        sbt.raygenRecord = d_raygen_record.d_ptr();
        sbt.missRecordBase = d_miss_record.d_ptr();
        sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof( oprt::MissRecord ) );
        sbt.missRecordCount = RAY_TYPE_COUNT;

        // HitGroup programs
        scene.create_hitgroup_programs( optix_context, module );
        std::vector<oprt::ProgramGroup> hitgroup_programs = scene.hitgroup_programs();
        std::copy( hitgroup_programs.begin(), hitgroup_programs.end(), std::back_inserter( program_groups ) );

        // Create sbts and bind it to hitgroup programs.
        scene.create_hitgroup_sbt( sbt );

        // Callable programs for sampling bsdf properties.
        std::vector<oprt::ProgramGroup> callable_programs;
        std::vector<oprt::CallableRecord> callable_records;
        create_material_sample_programs( optix_context, module, callable_programs, callable_records );
        create_texture_eval_programs( optix_context, module, callable_programs, callable_records );
        oprt::CUDABuffer<oprt::CallableRecord> d_callable_records;
        d_callable_records.alloc_copy(callable_records);
        sbt.callablesRecordBase = d_callable_records.d_ptr();
        sbt.callablesRecordCount = static_cast<unsigned int>( callable_records.size() );
        sbt.callablesRecordStrideInBytes = static_cast<unsigned int>( sizeof(oprt::CallableRecord ) ); 

        std::copy( callable_programs.begin(), callable_programs.end(), std::back_inserter( program_groups ) );

        // Create pipeline
        pipeline.create( optix_context, program_groups );
        // Initialize pipeline launch parameters.
        initLaunchParams( ias_handle, stream, params, d_params );

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "Path tracer", params.width, params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &params );

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                        output_buffer_type,
                        params.width,
                        params.height
                        );

                output_buffer.setStream( stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, params, camera );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, params, d_params, stream, pipeline, sbt );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    params.width,
                    params.height
                    );

            handleCameraUpdate( params, camera );
            handleResize( output_buffer, params );

            int num_launches = scene.num_samples() / scene.samples_per_launch();
            oprt::Assert(scene.num_samples()%scene.samples_per_launch() == 0, 
                "The number of total samples must be a multiple of the number of sampler per launch.");

            auto start_time = std::chrono::steady_clock::now();
            for (int i=0; i<num_launches; i++) {
                launchSubframe( output_buffer, params, d_params, stream, pipeline, sbt );
                params.subframe_index = i;
            }
            auto render_time = std::chrono::steady_clock::now() - start_time;
            std::chrono::minutes m = std::chrono::duration_cast<std::chrono::minutes>(render_time);
            std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(render_time);
            std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(render_time);
            printf("Render time: %dm %ds %dms\n", (int)m.count(), (int)s.count(), (int)ms.count());

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }
        /**
         * \brief Cleanup optix objects.
         */
        pipeline.destroy();
        module.destroy();
        for ( auto& pg : program_groups ) OPTIX_CHECK( optixProgramGroupDestroy(pg) );
        OPTIX_CHECK( optixDeviceContextDestroy( optix_context ) );
        oprt::cuda_frees(sbt.raygenRecord, sbt.missRecordBase, sbt.hitgroupRecordBase, 
                       params.accum_buffer,
                       d_params);

    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}