#include <glad/glad.h>

#include <GLFW/glfw3.h>

// Header file describe the scene
#include "scene_config.h"
#include "oprt.h"

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

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

void streamProgress(int current, int max, float elapsed_time, int progress_length)
{
    std::cout << "\rRendering: [";
    int progress = static_cast<int>(((float)(current+1) / max) * progress_length);
    for (int i = 0; i < progress; i++)
        std::cout << "+";
    for (int i = 0; i < progress_length - progress; i++)
        std::cout << " ";
    std::cout << "]";

    std::cout << " [" << std::fixed << std::setprecision(2) << elapsed_time << "s]";

    float percent = (float)(current) / max;
    std::cout << " (" << std::fixed << std::setprecision(2) << (float)(percent * 100.0f) << "%, ";
    std::cout << "Samples: " << current + 1 << " / " << max << ")" << std::flush;
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    Params params = {};
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
            outfile = pathJoin(OPRT_ROOT_DIR, argv[++i]).string();
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
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

        OPTIX_CHECK(optixInit());
        Context context;
        context.setDeviceId(0);
        context.create();

        // Load the scene
        Scene scene = my_scene();

        params.width                             = scene.film()->width();
        params.height                            = scene.film()->height();
        params.samples_per_launch                = scene.samplesPerLaunch();
        params.max_depth                         = scene.depth();

        // Initialize camera state.
        auto camera = scene.camera();
        initCameraState(camera);

        // Create instances that manage GAS, sbtOffset, and transform of geometries.
        std::vector<OptixInstance> instances;
        unsigned int sbt_base_offset = 0;
        unsigned int instance_id = 0;
        std::vector<AccelData*> accels;
        for ( auto ps : scene.primitiveInstances() ) {
            accels.push_back( new AccelData() );
            buildGas( context, *accels.back(), ps );
            buildInstances( context, *accels.back(), ps, sbt_base_offset, instance_id, instances ); 
        }

        CUDABuffer<OptixInstance> d_instances;
        d_instances.copyToDevice(instances);

        // Prepare build input for instances.
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances = d_instances.devicePtr();
        instance_input.instanceArray.numInstances = (unsigned int)instances.size();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            static_cast<OptixDeviceContext>(context), 
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
            static_cast<OptixDeviceContext>(context), 
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

        cuda_free(d_temp_buffer);
        d_instances.free();

        // createModule(context, pipeline_compile_options, ptx_module);
        // Prepare the pipeline
        std::string params_name = "params";
        Pipeline pipeline(params_name);
        pipeline.setDirectCallableDepth(4);   // The maximum call depth of direct callable programs.
        pipeline.setContinuationCallableDepth(4);   // The maximum call depth of continuation callable programs.
        pipeline.setNumPayloads(5);
        pipeline.setNumAttributes(5);
        // Create module
        Module module("optix/cuda/oprt.cu");
        module.create(context, pipeline.compileOptions());

        /**
         * \brief Create programs
         */
        std::vector<ProgramGroup> program_groups;
        // Raygen program
        auto raygen_program = createRayGenProgram( context, module, RG_FUNC_STR("raygen") );
        program_groups.push_back( raygen_program );
        // Create and bind sbt to raygen program
        CUDABuffer<RayGenRecord> d_raygen_record;
        RayGenRecord rg_record = {};
        raygen_program.bindRecord( &rg_record );
        d_raygen_record.copyToDevice( &rg_record, sizeof( RayGenRecord ) );
        
        // Miss program
        std::vector<ProgramGroup> miss_programs( RAY_TYPE_COUNT );
        miss_programs[0] = createMissProgram( context, module, MS_FUNC_STR("envmap") );
        miss_programs[1] = createMissProgram( context, Module(), nullptr );
        std::copy( miss_programs.begin(), miss_programs.end(), std::back_inserter( program_groups ) );
        // Create sbt for miss programs
        CUDABuffer<MissRecord> d_miss_record;
        MissRecord ms_records[RAY_TYPE_COUNT];
        scene.environment()->prepareData();
        for (int i=0; i<RAY_TYPE_COUNT; i++) {
            miss_programs[i].bindRecord( &ms_records[i] );
            ms_records[i].data.env_data = i == 0 ? scene.environment()->devicePtr() : nullptr;  
        }
        d_miss_record.copyToDevice( ms_records, sizeof(MissRecord) * RAY_TYPE_COUNT );

        // Bind sbts to raygen and miss program
        sbt.raygenRecord = d_raygen_record.devicePtr();
        sbt.missRecordBase = d_miss_record.devicePtr();
        sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof( MissRecord ) );
        sbt.missRecordCount = RAY_TYPE_COUNT;

        // HitGroup programs
        scene.createHitgroupPrograms( context, module );
        std::vector<ProgramGroup> hitgroup_programs = scene.hitgroupPrograms();
        std::copy( hitgroup_programs.begin(), hitgroup_programs.end(), std::back_inserter( program_groups ) );

        // Create sbts and bind it to hitgroup programs.
        scene.createHitgroupSBT( sbt );

        // Callable programs for sampling bsdf properties.
        std::vector<ProgramGroup> callable_programs;
        std::vector<CallableRecord> callable_records;
        createMaterialPrograms( context, module, callable_programs, callable_records );
        createTexturePrograms( context, module, callable_programs, callable_records );
        createEmitterPrograms( context, module, callable_programs, callable_records );
        CUDABuffer<CallableRecord> d_callable_records;
        d_callable_records.copyToDevice(callable_records);
        sbt.callablesRecordBase = d_callable_records.devicePtr();
        sbt.callablesRecordCount = static_cast<unsigned int>( callable_records.size() );
        sbt.callablesRecordStrideInBytes = static_cast<unsigned int>( sizeof(CallableRecord ) ); 

        std::copy( callable_programs.begin(), callable_programs.end(), std::back_inserter( program_groups ) );

        // Create pipeline
        pipeline.create( context, program_groups );
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

            int num_launches = scene.numSamples() / scene.samplesPerLaunch();
            Assert(scene.numSamples() % scene.samplesPerLaunch() == 0, 
                "The number of total samples must be a multiple of the number of sampler per launch.");

            auto start_time = std::chrono::steady_clock::now();
            for (int i=0; i<num_launches; i++) {
                launchSubframe( output_buffer, params, d_params, stream, pipeline, sbt );
                params.subframe_index = i;
                std::chrono::duration<float> elapsed_time = std::chrono::steady_clock::now() - start_time;
                streamProgress(i * scene.samplesPerLaunch() + scene.samplesPerLaunch() - 1, scene.numSamples(), elapsed_time.count(), 20);
            }
            std::cout << std::endl;
            auto render_time = std::chrono::steady_clock::now() - start_time;
            std::chrono::minutes m = std::chrono::duration_cast<std::chrono::minutes>(render_time);
            std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(render_time);
            std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(render_time);
            // Message(MSG_NORMAL, std::format("Render time: {}m {}s {}ms", (int)m.count(), (int)s.count(), (int)ms.count() - (int)(s.count()*1000)) );
            printf("Render time: %dm %ds %dms\n", (int)m.count(), (int)s.count(), (int)ms.count() - (int)(s.count()*1000));
            
            scene.film()->fillData(reinterpret_cast<Bitmap::Type*>(output_buffer.getHostPointer()), 
                scene.film()->width(), scene.film()->height(), 0, 0);
            scene.film()->write(outfile);

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
        for ( auto& pg : program_groups ) OPTIX_CHECK( optixProgramGroupDestroy(static_cast<OptixProgramGroup>(pg)) );
        OPTIX_CHECK( optixDeviceContextDestroy( static_cast<OptixDeviceContext>(context) ) );
        cuda_frees(sbt.raygenRecord, sbt.missRecordBase, sbt.hitgroupRecordBase, 
                       params.accum_buffer,
                       d_params);
        scene.cleanUp();
    }

    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}