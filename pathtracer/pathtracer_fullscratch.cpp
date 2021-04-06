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

#define SUCCEEDED 1

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

void initLaunchParams( 
    const OptixTraversableHandle& gas_handle,
    CUstream& stream,
    pt::Params& params,
    CUdeviceptr& d_params
)
{
    CUDA_CHECK (cudaMalloc(
        reinterpret_cast<void**>( &params.accum_buffer ),
        params.width * params.height * sizeof( float4 )
    ));
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.samples_per_launch = samples_per_launch;
    params.subframe_index = 0u;

    params.max_depth = 10;

    params.handle         = gas_handle;

    CUDA_CHECK( cudaStreamCreate( &stream) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( pt::Params ) ) );
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

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer,   // output buffer
                     pt::Params& params,                                // Launch parameters of OptiX
                     const CUdeviceptr& d_params,                       // Device side pointer of launch params
                     const CUstream& stream,                            // CUDA stream
                     const OptixPipeline& pipeline,                     // Pipeline of OptiX
                     OptixShaderBindingTable& sbt                 // Shader binding table 
) {
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

void initCameraState()
{
    camera.setEye(make_float3(278.0f, 273.0f, -900.0f));
    camera.setLookat(make_float3(278.0f, 273.0f, 330.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(35.0f);
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

void createContext( 
    OptixDeviceContext& context
)
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );
}

/****************************************************************
 * \brief Create geometry acceleration structure
 ****************************************************************/
void buildMeshAccel(
    const OptixDeviceContext& context,
    const std::vector<pt::Primitive>& primitives, 
    std::vector<CUdeviceptr>& d_vertices, 
    std::vector<CUdeviceptr>& d_indices, 
    std::vector<CUdeviceptr>& d_normals, 
    OptixTraversableHandle& gas_handle, 
    CUdeviceptr& d_gas_output_buffer )
{
    std::vector<OptixBuildInput> triangle_inputs(primitives.size());
    d_vertices.resize(primitives.size());
    d_indices.resize(primitives.size());
    d_normals.resize(primitives.size());
    for ( size_t meshID = 0; meshID < primitives.size(); meshID++ )
    {
        auto mesh = ((pt::TriangleMesh*)primitives[meshID].shape());
        // alloc and copy vertices data
        const size_t vertices_size_in_bytes = mesh->vertices().size() * sizeof(float3);
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_vertices[meshID]), vertices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>(d_vertices[meshID]),
            mesh->vertices().data(), vertices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        // alloc and copy indices data
        const size_t indices_size_in_bytes = mesh->indices().size() * sizeof(int3);
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_indices[meshID]), indices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>(d_indices[meshID]),
            mesh->indices().data(), indices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        // alloc and copy normals data
        if (!mesh->normals().empty())
        {
            const size_t normals_size_in_bytes = mesh->normals().size() * sizeof(float3);
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_normals[meshID]), normals_size_in_bytes ) );
            CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>(d_normals[meshID]),
                mesh->normals().data(), normals_size_in_bytes,
                cudaMemcpyHostToDevice
            ));
        }

        // alloc and copy material data
        CUdeviceptr d_mat_indices = 0;
        std::vector<uint32_t> mat_indices( mesh->indices().size(), meshID );
        const size_t mat_indices_size_in_bytes = mat_indices.size() * sizeof(uint32_t);
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>(d_mat_indices),
            mat_indices.data(),
            mat_indices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        unsigned int* triangle_input_flags = new unsigned int[1];
        triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        triangle_inputs[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_inputs[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_inputs[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_inputs[meshID].triangleArray.numVertices = static_cast<uint32_t>(mesh->vertices().size());
        triangle_inputs[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];
        triangle_inputs[meshID].triangleArray.flags = triangle_input_flags;
        triangle_inputs[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_inputs[meshID].triangleArray.indexStrideInBytes = sizeof(int3);
        triangle_inputs[meshID].triangleArray.numIndexTriplets = static_cast<uint32_t>(mesh->indices().size());
        triangle_inputs[meshID].triangleArray.indexBuffer = d_indices[meshID];
        triangle_inputs[meshID].triangleArray.numSbtRecords = 1;
        triangle_inputs[meshID].triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
        triangle_inputs[meshID].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        triangle_inputs[meshID].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        triangle_inputs.data(), 
        static_cast<int>(triangle_inputs.size()),
        &gas_buffer_sizes
    ));

    // temporarily buffer to build AS
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
        context,
        0,
        &accel_options,
        triangle_inputs.data(),
        triangle_inputs.size(),
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1
    ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    } 
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

/****************************************************************
 * \brief Create module
 ****************************************************************/
void createModule( 
    const OptixDeviceContext& context,
    OptixPipelineCompileOptions& pipeline_compile_options,
    OptixModule& module
)
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues      = 5;
    pipeline_compile_options.numAttributeValues    = 5;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optix/pathtracer.cu" );

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ) );
}

/****************************************************************
 * \brief Create program gruops
 ****************************************************************/
void createProgramGroups( 
    const OptixDeviceContext& context,
    const OptixModule& module,
    OptixProgramGroup& raygen_prog_group,
    OptixProgramGroup& miss_radiance_prog_group, 
    OptixProgramGroup& miss_occlusion_prog_group, 
    OptixProgramGroup& hitgroup_radiance_prog_group, 
    OptixProgramGroup& hitgroup_occlusion_prog_group
)
{
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        OptixProgramGroupDesc raygen_prog_group_desc    = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = RG_FUNC_STR("raygen");

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &raygen_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &raygen_prog_group
                    ) );
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = MS_FUNC_STR("radiance");
        sizeof_log                                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &miss_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &miss_radiance_prog_group
                    ) );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log                                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &miss_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &miss_occlusion_prog_group
                    ) );
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc        = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = CH_FUNC_STR("mesh");
        sizeof_log                                       = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &hit_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &hitgroup_radiance_prog_group
                    ) );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = CH_FUNC_STR("occlusion");
        sizeof_log                                       = sizeof( log );
        OPTIX_CHECK( optixProgramGroupCreate(
                    context,
                    &hit_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &hitgroup_occlusion_prog_group
                    ) );
    }
}

/****************************************************************
 * \brief Create pipeline
 ****************************************************************/
void createPipeline( 
    const OptixDeviceContext& context,
    const OptixPipelineCompileOptions& pipeline_compile_options,
    const std::vector<OptixProgramGroup>& program_groups,
    OptixPipeline& pipeline
)
{

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 2;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups.data(),
                program_groups.size(),
                log,
                &sizeof_log,
                &pipeline
                ) );

    OptixStackSizes stack_size = {};
    for (auto& pg : program_groups) {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( pg, &stack_size ) );
    }

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes(
                &stack_size,
                max_trace_depth,
                max_cc_depth,
                max_dc_depth,
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size
                ) );

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK( optixPipelineSetStackSize(
                pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth
                ) );
}

/****************************************************************
 * \brief Create shader binding table
 ****************************************************************/
void createSBT( 
    const OptixDeviceContext& context, 
    const std::vector<OptixProgramGroup>& program_groups,
    const std::vector<pt::Primitive> primitives,
    const std::vector<CUdeviceptr>& d_vertices,
    const std::vector<CUdeviceptr>& d_indices, 
    const std::vector<CUdeviceptr>& d_normals,
    OptixShaderBindingTable& sbt
)
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( pt::RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );

    pt::RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( program_groups[0], &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );


    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof( pt::MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size * RAY_TYPE_COUNT ) );

    pt::MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( program_groups[1],  &ms_sbt[0] ) );
    ms_sbt[0].data.bg_color = make_float4( 0.0f );
    OPTIX_CHECK( optixSbtRecordPackHeader( program_groups[2], &ms_sbt[1] ) );
    ms_sbt[1].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss_records ),
                ms_sbt,
                miss_record_size*RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice
                ) );

    const size_t hitgroup_record_size = sizeof(pt::HitGroupRecord);
    std::vector<pt::HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * primitives.size());

    for (size_t meshID = 0; meshID < primitives.size(); meshID++) 
    {
        {
            const int sbt_idx = meshID * RAY_TYPE_COUNT + 0;

            OPTIX_CHECK( optixSbtRecordPackHeader( program_groups[3], &hitgroup_records[sbt_idx] ) );
            // switch ( primitives[meshID].material()->type() )
            // {
            // case pt::MaterialType::Diffuse: {
            //     hitgroup_records[sbt_idx].data.albedo = make_float3(1.0f);
            //     hitgroup_records[sbt_idx].data.emission = make_float3(0.0f);
            //     break;
            // }
            // case pt::MaterialType::Emitter: {
            //     hitgroup_records[sbt_idx].data.albedo = make_float3(0.0f);
            //     hitgroup_records[sbt_idx].data.emission = make_float3(1.0f);
            //     break;
            // }
            // }

            // hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float3*>(d_vertices[meshID]);
            // hitgroup_records[sbt_idx].data.normals = reinterpret_cast<float3*>(d_normals[meshID]);
            // hitgroup_records[sbt_idx].data.indices = reinterpret_cast<int3*>(d_indices[meshID]);
            hitgroup_records[sbt_idx].data.shapedata = reinterpret_cast<void*>(primitives[meshID].shape()->get_dptr());
            hitgroup_records[sbt_idx].data.matptr = (primitives[meshID].material()->get_dptr());
        }

        {
            const int sbt_idx = meshID * RAY_TYPE_COUNT + 1;
            memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
            OPTIX_CHECK( optixSbtRecordPackHeader( program_groups[4], &hitgroup_records[sbt_idx] ) );
        }
    }

    pt::CUDABuffer<pt::HitGroupRecord> d_hitgroup_records;
    d_hitgroup_records.alloc_copy(hitgroup_records);

    sbt.raygenRecord                = d_raygen_record;
    sbt.missRecordBase              = d_miss_records;
    sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    sbt.missRecordCount             = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase          = d_hitgroup_records.d_ptr();
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    sbt.hitgroupRecordCount         = hitgroup_records.size();
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    pt::Params params = {};
    CUdeviceptr d_params;

    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    // Prepare the states for launching ray tracer
    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr d_gas_output_buffer = 0;

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
        initCameraState();

        CUDA_CHECK(cudaFree(0));

        OptixDeviceContext optix_context;
        CUcontext cu_context = 0;
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cu_context, &options, &optix_context));

        pt::Scene scene = my_scene();

        params.width                             = scene.width();
        params.height                            = scene.height();
        params.samples_per_launch                = scene.samples_per_launch();
        params.max_depth                         = scene.depth();

#ifndef SUCCEEDED
        std::vector<OptixInstance> instances;
        unsigned int sbt_base_offset = 0; 
        unsigned int instance_id = 0;
        for (auto& ps : scene.primitive_instances()) {
            pt::AccelData accel = {};
            pt::build_gas(optix_context, accel, ps);
            pt::build_instances(optix_context, accel, ps, sbt_base_offset, instance_id, instances);
        }

        CUdeviceptr d_instances;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), sizeof(OptixInstance)*instances.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances),
            instances.data(), instances.size() * sizeof(OptixInstance), 
            cudaMemcpyHostToDevice));

        // Prepare build input for instances.
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances = d_instances;
        instance_input.instanceArray.numInstances = (unsigned int) instances.size();

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

        pt::cuda_frees(d_temp_buffer, d_instances);
#else
        pt::AccelData accel = {};
        pt::build_gas(optix_context, accel, scene.primitive_instances()[0]);
#endif

        std::vector<CUdeviceptr> d_vertices;
        std::vector<CUdeviceptr> d_normals;
        std::vector<CUdeviceptr> d_indices;

        for (auto& p : scene.primitive_instances()[0].primitives()) {
            auto mesh = ((pt::TriangleMesh*)p.shape());
            d_vertices.emplace_back(mesh->get_dvertices());
            d_normals.emplace_back(mesh->get_dnormals());
            d_indices.emplace_back(mesh->get_dindices());
        }

        // createModule(optix_context, pipeline_compile_options, ptx_module);
        // Prepare the pipeline
        std::string params_name = "params";
        pt::Pipeline pt_pipeline(params_name);
        // Create module
        pt::Module pt_module("optix/pathtracer.cu");
        pt_module.create(optix_context, pt_pipeline.compile_options());

        /**
         * \brief Create programs
         */
        std::vector<OptixProgramGroup> program_groups;
        // Raygen program
        pt::ProgramGroup raygen_program(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
        raygen_program.create( optix_context, pt::ProgramEntry( (OptixModule)pt_module, RG_FUNC_STR("raygen") ) );
        program_groups.push_back( (OptixProgramGroup)raygen_program );
        // Create and bind sbt for raygen program
        pt::CUDABuffer<pt::RayGenRecord> d_raygen_record;
        pt::RayGenRecord rg_record = {};
        OPTIX_CHECK( optixSbtRecordPackHeader( (OptixProgramGroup)raygen_program, &rg_record ) );
        d_raygen_record.alloc_copy( &rg_record, sizeof( pt::RayGenRecord ) );

        // Miss program
        std::vector<pt::ProgramGroup> miss_programs( RAY_TYPE_COUNT, pt::ProgramGroup( OPTIX_PROGRAM_GROUP_KIND_MISS ) );
        miss_programs[0].create( optix_context, pt::ProgramEntry( (OptixModule)pt_module, MS_FUNC_STR("radiance") ) );
        miss_programs[1].create( optix_context, pt::ProgramEntry( nullptr, nullptr ) );
        std::copy( miss_programs.begin(), miss_programs.end(), std::back_inserter( program_groups ) );
        // Create sbt for miss programs
        pt::CUDABuffer<pt::MissRecord> d_miss_record;
        pt::MissRecord ms_records[RAY_TYPE_COUNT];
        for (int i=0; i<RAY_TYPE_COUNT; i++) {
            OPTIX_CHECK( optixSbtRecordPackHeader( (OptixProgramGroup)miss_programs[i], &ms_records[i] ) );
            ms_records[i].data.bg_color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        d_miss_record.alloc_copy( ms_records, sizeof(pt::MissRecord)*RAY_TYPE_COUNT );

        // // Attach sbts for raygen and miss program
        sbt.raygenRecord = d_raygen_record.d_ptr();
        sbt.missRecordBase = d_miss_record.d_ptr();
        sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(pt::MissRecord));
        sbt.missRecordCount = RAY_TYPE_COUNT;

        // HitGroup programs
        scene.create_hitgroup_programs( optix_context, (OptixModule)pt_module );
        std::vector<pt::ProgramGroup> hitgroup_programs = scene.hitgroup_programs();
        std::copy(hitgroup_programs.begin(), hitgroup_programs.end(), std::back_inserter( program_groups ) );

        scene.create_hitgroup_sbt( (OptixModule)pt_module, sbt );

        pt_pipeline.create( optix_context, program_groups );

        initLaunchParams( accel.meshes.handle, stream, params, d_params );

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

                    updateState( output_buffer, params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, params, d_params, stream, (OptixPipeline)pt_pipeline, sbt );
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

            handleCameraUpdate( params );
            handleResize( output_buffer, params );
            launchSubframe( output_buffer, params, d_params, stream, (OptixPipeline)pt_pipeline, sbt );

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
        // OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        pt_pipeline.destroy();
        for ( auto& pg : program_groups ) OPTIX_CHECK( optixProgramGroupDestroy(pg) );
        // OPTIX_CHECK( optixModuleDestroy( ptx_module ) );
        pt_module.destroy();
        OPTIX_CHECK( optixDeviceContextDestroy( optix_context ) );
        pt::cuda_frees(sbt.raygenRecord, sbt.missRecordBase, sbt.hitgroupRecordBase, 
                       d_gas_output_buffer, 
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