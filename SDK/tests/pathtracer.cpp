//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


// TODO:
//   - Try dividing vertices data to cornel box and bunny.
//     - Maybe, I have to add diffuse color data for Model struct.

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

#include <core/util.h>
#include <core/pathtracer.h>
#include "src/shape/trianglemesh.h"

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state 
int32_t mouse_button = -1;
int32_t samples_per_launch = 4;

template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord = Record<RayGenData>;
using MissRecord = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;

struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};

struct Instance
{
    float transform[12];
};

// This should be set to global scope.
struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle gas_handle = 0; // Traversable handle for triangle AS
    CUdeviceptr d_gas_output_buffer = 0; // Triangle AS memory
    
    std::vector<CUdeviceptr> d_vertices;
    std::vector<CUdeviceptr> d_indices;
    std::vector<CUdeviceptr> d_normals;
    std::vector<CUdeviceptr> d_materials;

    // 
    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup radiance_miss_prog_group = 0;
    OptixProgramGroup occlusion_miss_prog_group = 0;
    OptixProgramGroup radiance_dielectric_prog_group = 0;
    OptixProgramGroup occlusion_dielectric_prog_group = 0;
    OptixProgramGroup radiance_diffuse_prog_group = 0;
    OptixProgramGroup occlusion_diffuse_prog_group = 0;
    OptixProgramGroup radiance_metal_prog_group = 0;
    OptixProgramGroup occlusion_metal_prog_group = 0;
    OptixProgramGroup radiance_emission_prog_group = 0;
    OptixProgramGroup occlusion_emission_prog_group = 0;

    // 
    OptixModule ptx_module = 0;
    OptixModule bsdf_module = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;

    CUstream stream = 0;
    Params params;
    Params* d_params;

    OptixShaderBindingTable sbt = {};
};

// scene data
const int32_t TRIANGLE_COUNT = 12;
const int32_t MAT_COUNT = 5;

std::vector<TriangleMesh> meshes;
std::vector<Material*> materials;
std::vector<MaterialType> mat_types;

void initTriangleMeshes()
{
    // Floor ------------------------------------
    std::vector<float3> floor_vertices;
    std::vector<float3> floor_normals(6, make_float3(0.0f, 1.0f, 0.0f));
    std::vector<int3> floor_indices;
    floor_vertices.emplace_back(make_float3(  0.0f, 0.0f,   0.0f));
    floor_vertices.emplace_back(make_float3(  0.0f, 0.0f, 559.2f));
    floor_vertices.emplace_back(make_float3(556.0f, 0.0f, 559.2f));
    floor_vertices.emplace_back(make_float3(  0.0f, 0.0f,   0.0f));
    floor_vertices.emplace_back(make_float3(556.0f, 0.0f, 559.2f));
    floor_vertices.emplace_back(make_float3(556.0f, 0.0f,   0.0f));
    floor_indices.emplace_back(make_int3(0, 1, 2));
    floor_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh floor_mesh(floor_vertices, floor_indices, floor_normals);
    meshes.emplace_back(floor_mesh);
    materials.emplace_back(new Diffuse(make_float3(0.8f)));

    // Ceiling ------------------------------------
    std::vector<float3> ceiling_vertices;
    std::vector<float3> ceiling_normals(6, make_float3(0.0f, -1.0f, 0.0f));
    std::vector<int3> ceiling_indices;
    ceiling_vertices.emplace_back(make_float3(  0.0f, 548.8f,   0.0f));
    ceiling_vertices.emplace_back(make_float3(556.0f, 548.8f,   0.0f));
    ceiling_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    ceiling_vertices.emplace_back(make_float3(  0.0f, 548.8f,   0.0f));
    ceiling_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    ceiling_vertices.emplace_back(make_float3(  0.0f, 548.8f, 559.2f));
    ceiling_indices.emplace_back(make_int3(0, 1, 2));
    ceiling_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh ceiling_mesh(ceiling_vertices, ceiling_indices, ceiling_normals);
    meshes.emplace_back(ceiling_mesh);
    materials.emplace_back(new Diffuse(make_float3(0.8f)));

    // Back wall ------------------------------------
    std::vector<float3> back_wall_vertices;
    std::vector<float3> back_wall_normals(6, make_float3(0.0f, 0.0f, -1.0f));
    std::vector<int3> back_wall_indices;
    back_wall_vertices.emplace_back(make_float3(  0.0f,   0.0f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(  0.0f, 548.8f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(  0.0f,   0.0f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(556.0f,   0.0f, 559.2f));
    back_wall_indices.emplace_back(make_int3(0, 1, 2));
    back_wall_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh back_wall_mesh(back_wall_vertices, back_wall_indices, back_wall_normals);
    meshes.emplace_back(back_wall_mesh);
    materials.emplace_back(new Diffuse(make_float3(0.8f)));

    // Right wall ------------------------------------
    std::vector<float3> right_wall_vertices;
    std::vector<float3> right_wall_normals(6, make_float3(1.0f, 0.0f, 0.0f));
    std::vector<int3> right_wall_indices;
    right_wall_vertices.emplace_back(make_float3(0.0f,   0.0f,   0.0f));
    right_wall_vertices.emplace_back(make_float3(0.0f, 548.8f,   0.0f));
    right_wall_vertices.emplace_back(make_float3(0.0f, 548.8f, 559.2f));
    right_wall_vertices.emplace_back(make_float3(0.0f,   0.0f,   0.0f));
    right_wall_vertices.emplace_back(make_float3(0.0f, 548.8f, 559.2f));
    right_wall_vertices.emplace_back(make_float3(0.0f,   0.0f, 559.2f));
    right_wall_indices.emplace_back(make_int3(0, 1, 2));
    right_wall_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh right_wall_mesh(right_wall_vertices, right_wall_indices, right_wall_normals);
    meshes.emplace_back(right_wall_mesh);
    materials.emplace_back(new Diffuse(make_float3(0.05f, 0.8f, 0.05f)));

    // Left wall ------------------------------------
    std::vector<float3> left_wall_vertices;
    std::vector<float3> left_wall_normals(6, make_float3(-1.0f, 0.0f, 0.0f));
    std::vector<int3> left_wall_indices;
    left_wall_vertices.emplace_back(make_float3(556.0f,   0.0f,   0.0f));
    left_wall_vertices.emplace_back(make_float3(556.0f,   0.0f, 559.2f));
    left_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    left_wall_vertices.emplace_back(make_float3(556.0f,   0.0f,   0.0f));
    left_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    left_wall_vertices.emplace_back(make_float3(556.0f, 548.8f,   0.0f));
    left_wall_indices.emplace_back(make_int3(0, 1, 2));
    left_wall_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh left_wall_mesh(left_wall_vertices, left_wall_indices, left_wall_normals);
    meshes.emplace_back(left_wall_mesh);
    materials.emplace_back(new Diffuse(make_float3(0.8f, 0.05f, 0.05f)));

    // Ceiling light ------------------------------------
    std::vector<float3> ceiling_light_vertices;
    std::vector<float3> ceiling_light_normals(6, make_float3(0.0f, -1.0f, 0.0f));
    std::vector<int3> ceiling_light_indices;
    ceiling_light_vertices.emplace_back(make_float3(343.0f, 548.6f, 227.0f));
    ceiling_light_vertices.emplace_back(make_float3(213.0f, 548.6f, 227.0f));
    ceiling_light_vertices.emplace_back(make_float3(213.0f, 548.6f, 332.0f));
    ceiling_light_vertices.emplace_back(make_float3(343.0f, 548.6f, 227.0f));
    ceiling_light_vertices.emplace_back(make_float3(213.0f, 548.6f, 332.0f));
    ceiling_light_vertices.emplace_back(make_float3(343.0f, 548.6f, 332.0f));
    ceiling_light_indices.emplace_back(make_int3(0, 1, 2));
    ceiling_light_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh ceiling_light_mesh(ceiling_light_vertices, ceiling_light_indices, ceiling_light_normals);
    meshes.emplace_back(ceiling_light_mesh);
    materials.emplace_back(new Emission(make_float3(15.0f)));

    float3 cornel_center = make_float3(556.0f / 2.0f, 548.0f / 2.0f, 559.2f / 2.0f);
    // Small light ------------------------------------
    /*std::vector<Vertex> small_light_vertices;
    std::vector<Normal> small_light_normals(6, Normal(0.0f, 0.0f, 1.0f, 0.0f));
    std::vector<int3> small_light_indices;
    float small_light_size = 50.0f;
    small_light_vertices.emplace_back(
        cornel_center.x - small_light_size / 2.0f,
        cornel_center.y - small_light_size / 2.0f,
        cornel_center.z, 0.0f);
    small_light_vertices.emplace_back(
        cornel_center.x - small_light_size / 2.0f,
        cornel_center.y + small_light_size / 2.0f,
        cornel_center.z, 0.0f);
    small_light_vertices.emplace_back(
        cornel_center.x + small_light_size / 2.0f,
        cornel_center.y - small_light_size / 2.0f,
        cornel_center.z, 0.0f);
    small_light_vertices.emplace_back(
        cornel_center.x + small_light_size / 2.0f,
        cornel_center.y - small_light_size / 2.0f,
        cornel_center.z, 0.0f);
    small_light_vertices.emplace_back(
        cornel_center.x - small_light_size / 2.0f,
        cornel_center.y + small_light_size / 2.0f,
        cornel_center.z, 0.0f);
    small_light_vertices.emplace_back(
        cornel_center.x + small_light_size / 2.0f,
        cornel_center.y + small_light_size / 2.0f,
        cornel_center.z, 0.0f);
    small_light_indices.emplace_back(make_int3(0, 1, 2));
    small_light_indices.emplace_back(make_int3(3, 4, 5));
    TriangleMesh small_light_mesh(small_light_vertices, small_light_indices, small_light_normals);
    meshes.emplace_back(small_light_mesh);
    materials.emplace_back(new Emission(make_float3(5.0f, 0.1f, 0.1f)));*/

    // Bunny ------------------------------------
    //TriangleMesh bunny("../../model/bunny.obj",                         // filename
    //    cornel_center,       // center position
    //    150.0f,                                                         // size
    //    make_float3(1, 1, -1));                                         // axis
    //meshes.emplace_back(bunny);
    //materials.emplace_back(new Dielectric(make_float3(1.0f), 1.52f));

    // MMAPs ------------------------------------
    /*TriangleMesh mmaps_glass("../../model/mmaps_glass.obj",
        cornel_center,
        2.0f,
        make_float3(1, 1, 1), false);
    meshes.emplace_back(mmaps_glass);
    materials.emplace_back(new Dielectric(make_float3(1.0f), 1.52f));

    TriangleMesh mmaps_mirror("../../model/mmaps.obj",
        cornel_center,
        2.0f,
        make_float3(1, 1, 1), false);
    meshes.emplace_back(mmaps_mirror);
    materials.emplace_back(new Metal(make_float3(1.0f, 1.0f, 1.0f), 1.0f));*/
}

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
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ));

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

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ));
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

void initLaunchParams( PathTracerState& state )
{
    CUDA_CHECK (cudaMalloc(
        reinterpret_cast<void**>( &state.params.accum_buffer ),
        state.params.width * state.params.height * sizeof( float4 )
    ));
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;

    state.params.max_depth = 10;

    state.params.handle         = state.gas_handle;

    CUDA_CHECK( cudaStreamCreate( &state.stream) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
}

void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
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

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast<void*>( state.d_params ),
        &state.params, sizeof( Params ),
        cudaMemcpyHostToDevice, state.stream
    ));

    // Start ray tracing on device.
    OPTIX_CHECK( optixLaunch(
        state.pipeline, 
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ),
        &state.sbt,
        state.params.width,
        state.params.height,
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

// ------------------------------------------------------------------------------------------------------------
// Initialize camera state
// ------------------------------------------------------------------------------------------------------------
void initCameraState()
{
    camera.setEye( make_float3(278.0f, 273.0f, -900.0f ) );
    camera.setLookat( make_float3( 278.0f, 273.0f, 330.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 35.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame(
        make_float3( 1.0f, 0.0f, 0.0f ),
        make_float3( 0.0f, 0.0f, 1.0f ),
        make_float3( 0.0f, 1.0f, 0.0f )
    );
    trackball.setGimbalLock( true );
}

// ------------------------------------------------------------------------------------------------------------
// Create context
// ------------------------------------------------------------------------------------------------------------
void createContext( PathTracerState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext cu_ctx = 0;
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}

// ------------------------------------------------------------------------------------------------------------
// Build Acceleration structure for mesh
// ------------------------------------------------------------------------------------------------------------
void buildMeshAccel( PathTracerState& state )
{
    std::vector<OptixBuildInput> triangle_inputs(meshes.size());
    state.d_vertices.resize(meshes.size());
    state.d_indices.resize(meshes.size());
    state.d_normals.resize(meshes.size());
    int mesh_offset = 0;
    for (int meshID = 0; meshID < meshes.size(); meshID++)
    {
        // alloc and copy vertices data
        const size_t vertices_size_in_bytes = meshes[meshID].vertices.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices[meshID]), vertices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.d_vertices[meshID]),
            meshes[meshID].vertices.data(), vertices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        // alloc and copy indices data
        const size_t indices_size_in_bytes = meshes[meshID].indices.size() * sizeof(int3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_indices[meshID]), indices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.d_indices[meshID]),
            meshes[meshID].indices.data(), indices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        // alloc and copy normals data
        if (!meshes[meshID].normals.empty())
        {
            const size_t normals_size_in_bytes = meshes[meshID].normals.size() * sizeof(float3);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_normals[meshID]), normals_size_in_bytes));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(state.d_normals[meshID]),
                meshes[meshID].normals.data(), normals_size_in_bytes,
                cudaMemcpyHostToDevice
            ));
        }

        // alloc and copy material data
        CUdeviceptr d_mat_indices = 0;
        std::vector<uint32_t> mat_indices(meshes[meshID].indices.size(), meshID);
        const size_t mat_indices_size_in_bytes = mat_indices.size() * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_mat_indices),
            mat_indices.data(),
            mat_indices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        unsigned int* triangle_input_flags = new unsigned int[1];
        triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        // When undefined pointer is set to new pointer in local function, 
        // behaviour of this pointer will be undefined... maybe...
        triangle_inputs[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_inputs[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_inputs[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_inputs[meshID].triangleArray.numVertices = static_cast<uint32_t>(meshes[meshID].vertices.size());
        triangle_inputs[meshID].triangleArray.vertexBuffers = &state.d_vertices[meshID];
        triangle_inputs[meshID].triangleArray.flags = triangle_input_flags;
        triangle_inputs[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_inputs[meshID].triangleArray.indexStrideInBytes = sizeof(int3);
        triangle_inputs[meshID].triangleArray.numIndexTriplets = static_cast<uint32_t>(meshes[meshID].indices.size());
        triangle_inputs[meshID].triangleArray.indexBuffer = state.d_indices[meshID];
        triangle_inputs[meshID].triangleArray.numSbtRecords = 1;
        triangle_inputs[meshID].triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
        triangle_inputs[meshID].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        triangle_inputs[meshID].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // BUG: Invalid arguments error occured with `triangleArray.flags`. 
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
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
        state.context,
        0,
        &accel_options,
        triangle_inputs.data(),
        triangle_inputs.size(),
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty,
        1
    ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    } 
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

// ------------------------------------------------------------------------------------------------------------
// Create modules
// ------------------------------------------------------------------------------------------------------------
void createModule( PathTracerState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 5;
    state.pipeline_compile_options.numAttributeValues = 2;

#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    {
        const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "cuda/pathtracer.cu");
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.ptx_module
        ));
    }

    {
        const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "cuda/bsdf.cu");
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.bsdf_module
        ));
    }

}

// ------------------------------------------------------------------------------------------------------------
// Create radiance and occlusion program for dielectric material
// ------------------------------------------------------------------------------------------------------------
static void createDiffuseProgram(PathTracerState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup radiance_diffuse_prog_group;
    OptixProgramGroupOptions radiance_diffuse_prog_group_options = {};
    OptixProgramGroupDesc radiance_diffuse_prog_group_desc = {};
    radiance_diffuse_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_diffuse_prog_group_desc.hitgroup.moduleCH = state.bsdf_module;
    radiance_diffuse_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance__diffuse";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_diffuse_prog_group_desc,
        1,
        &radiance_diffuse_prog_group_options,
        log,
        &sizeof_log,
        &radiance_diffuse_prog_group
    ));

    program_groups.push_back(radiance_diffuse_prog_group);
    state.radiance_diffuse_prog_group = radiance_diffuse_prog_group;

    OptixProgramGroup occlusion_diffuse_prog_group;
    OptixProgramGroupOptions occlusion_diffuse_prog_group_options = {};
    OptixProgramGroupDesc occlusion_diffuse_prog_group_desc = {};
    occlusion_diffuse_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_diffuse_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    occlusion_diffuse_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_diffuse_prog_group_desc,
        1,
        &occlusion_diffuse_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_diffuse_prog_group));
    program_groups.push_back(occlusion_diffuse_prog_group);
    state.occlusion_diffuse_prog_group = occlusion_diffuse_prog_group;
}

// ------------------------------------------------------------------------------------------------------------
// Create radiance and occlusion program for dielectric material
// ------------------------------------------------------------------------------------------------------------
static void createDielectricProgram(PathTracerState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup radiance_dielectric_prog_group;
    OptixProgramGroupOptions radiance_dielectric_prog_group_options = {};
    OptixProgramGroupDesc radiance_dielectric_prog_group_desc = {};
    radiance_dielectric_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_dielectric_prog_group_desc.hitgroup.moduleCH = state.bsdf_module;
    radiance_dielectric_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance__dielectric";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_dielectric_prog_group_desc,
        1,
        &radiance_dielectric_prog_group_options,
        log,
        &sizeof_log,
        &radiance_dielectric_prog_group
    ));

    program_groups.push_back(radiance_dielectric_prog_group);
    state.radiance_dielectric_prog_group = radiance_dielectric_prog_group;

    OptixProgramGroup occlusion_dielectric_prog_group;
    OptixProgramGroupOptions occlusion_dielectric_prog_group_options = {};
    OptixProgramGroupDesc occlusion_dielectric_prog_group_desc = {};
    occlusion_dielectric_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_dielectric_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    occlusion_dielectric_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_dielectric_prog_group_desc,
        1,
        &occlusion_dielectric_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_dielectric_prog_group));
    program_groups.push_back(occlusion_dielectric_prog_group);
    state.occlusion_dielectric_prog_group = occlusion_dielectric_prog_group;
}

// ------------------------------------------------------------------------------------------------------------
// Create radiance and occlusion program for emission material
// ------------------------------------------------------------------------------------------------------------
static void createEmissionProgram(PathTracerState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup radiance_emission_prog_group;
    OptixProgramGroupOptions radiance_emission_prog_group_options = {};
    OptixProgramGroupDesc radiance_emission_prog_group_desc = {};
    radiance_emission_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_emission_prog_group_desc.hitgroup.moduleCH = state.bsdf_module;
    radiance_emission_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance__emission";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_emission_prog_group_desc,
        1,
        &radiance_emission_prog_group_options,
        log,
        &sizeof_log,
        &radiance_emission_prog_group
    ));

    program_groups.push_back(radiance_emission_prog_group);
    state.radiance_emission_prog_group = radiance_emission_prog_group;

    OptixProgramGroup occlusion_emission_prog_group;
    OptixProgramGroupOptions occlusion_emission_prog_group_options = {};
    OptixProgramGroupDesc occlusion_emission_prog_group_desc = {};
    occlusion_emission_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_emission_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    occlusion_emission_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_emission_prog_group_desc,
        1,
        &occlusion_emission_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_emission_prog_group));
    program_groups.push_back(occlusion_emission_prog_group);
    state.occlusion_emission_prog_group = occlusion_emission_prog_group;
}

// ------------------------------------------------------------------------------------------------------------
// Create radiance and occlusion program for metal material
// ------------------------------------------------------------------------------------------------------------
static void createMetalProgram(PathTracerState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup radiance_metal_prog_group;
    OptixProgramGroupOptions radiance_metal_prog_group_options = {};
    OptixProgramGroupDesc radiance_metal_prog_group_desc = {};
    radiance_metal_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_metal_prog_group_desc.hitgroup.moduleCH = state.bsdf_module;
    radiance_metal_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance__metal";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_metal_prog_group_desc,
        1,
        &radiance_metal_prog_group_options,
        log,
        &sizeof_log,
        &radiance_metal_prog_group
    ));

    program_groups.push_back(radiance_metal_prog_group);
    state.radiance_metal_prog_group = radiance_metal_prog_group;

    OptixProgramGroup occlusion_metal_prog_group;
    OptixProgramGroupOptions occlusion_metal_prog_group_options = {};
    OptixProgramGroupDesc occlusion_metal_prog_group_desc = {};
    occlusion_metal_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_metal_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    occlusion_metal_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_metal_prog_group_desc,
        1,
        &occlusion_metal_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_metal_prog_group));
    program_groups.push_back(occlusion_metal_prog_group);
    state.occlusion_metal_prog_group = occlusion_metal_prog_group;
}

// ------------------------------------------------------------------------------------------------------------
// Create Ray generation and miss programs
// ------------------------------------------------------------------------------------------------------------
static void createRaygenAndMissProgram( PathTracerState& state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions program_group_options = {};

    char log[2048];
    size_t sizeof_log = sizeof( log );

    {
        OptixProgramGroup raygen_prog_group;
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group
        ));
        program_groups.push_back(raygen_prog_group);
        state.raygen_prog_group = raygen_prog_group;
    }

    {
        OptixProgramGroup radiance_miss_prog_group;
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &radiance_miss_prog_group
        ));
        program_groups.push_back(radiance_miss_prog_group);
        state.radiance_miss_prog_group = radiance_miss_prog_group;

        OptixProgramGroup occlusion_miss_prog_group;
        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr;
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &occlusion_miss_prog_group
        ));
        program_groups.push_back(occlusion_miss_prog_group);
        state.occlusion_miss_prog_group = occlusion_miss_prog_group;
    }
}

// ------------------------------------------------------------------------------------------------------------
// Create pipeline object
// ------------------------------------------------------------------------------------------------------------
void createPipeline(PathTracerState& state)
{
    std::vector<OptixProgramGroup> program_groups;
    createDiffuseProgram(state, program_groups);
    createDielectricProgram(state, program_groups);
    createEmissionProgram(state, program_groups);
    createMetalProgram(state, program_groups);
    createRaygenAndMissProgram(state, program_groups);

    std::cerr << "program_groups.size(): " << program_groups.size() << std::endl;

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2; // depth setting to track ray
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    // ========== Memo : log string ==========
    // string is used to report information about any compilation that may have occured,
    // such as compile errors or verbose information about the compilation result.
    // To detect truncated output, the size of the log message is reported as an output
    // parameter. It is not recommended that you call the function again to get the full
    // output because this could result in unnecessary and lengthy work, or different 
    // output for cache hits. If an error occured, the information that would be reported
    // in the log string is also reported by the device context log callback (when provided).
    // ======================================

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        program_groups.size(), 
        log,
        &sizeof_log,
        &state.pipeline
    ));

    // We need to specify the max traversal depth. 
    // Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_miss_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_miss_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_diffuse_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_diffuse_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_dielectric_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_dielectric_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_metal_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_metal_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_emission_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_emission_prog_group, &stack_sizes));

    uint32_t max_trace_depth = 5;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

/** MEMO:
 * Shader binding table connects geometric data to programs and their parameters. */
// ------------------------------------------------------------------------------------------------------------
// Create shader binding table (SBT)
// ------------------------------------------------------------------------------------------------------------
void createSBT(PathTracerState& state)
{
    // For ray-generation program
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    /** MEMO:
     *  optixSbtRecordPackHeader() and a given OptixProgramGroup 
     *  object are used to fill the header of an SBT record. */

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    // For miss program
    CUdeviceptr d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_prog_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_miss_prog_group, &ms_sbt[1]));
    ms_sbt[1].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

    // For radiance and occlusion program
    // TODO: Copy Record object w.r.t each materials
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * meshes.size());
    for (int meshID = 0; meshID < meshes.size(); meshID++)
    {
        {
            const int sbt_idx = meshID * RAY_TYPE_COUNT + 0;

            switch (materials[meshID]->type()) {
            case MaterialType::Diffuse: {
                OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_diffuse_prog_group, &hitgroup_records[sbt_idx]));
                Diffuse* diffuse_data = (Diffuse*)materials[meshID];
                hitgroup_records[sbt_idx].data.shading.diffuse.mat_color = diffuse_data->mat_color;
                hitgroup_records[sbt_idx].data.shading.diffuse.is_normal = diffuse_data->is_normal;
                break;
            }
            case MaterialType::Dielectric: {
                OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_dielectric_prog_group, &hitgroup_records[sbt_idx]));
                Dielectric* dielectric_data = (Dielectric*)materials[meshID];
                hitgroup_records[sbt_idx].data.shading.dielectric.mat_color = dielectric_data->mat_color;
                hitgroup_records[sbt_idx].data.shading.dielectric.ior = dielectric_data->ior;
                break;
            }
            case MaterialType::Metal: {
                OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_metal_prog_group, &hitgroup_records[sbt_idx]));
                Metal* metal_data = (Metal*)materials[meshID];
                hitgroup_records[sbt_idx].data.shading.metal.mat_color = metal_data->mat_color;
                hitgroup_records[sbt_idx].data.shading.metal.reflection = metal_data->reflection;
                break;
            }
            case MaterialType::Emission: {
                OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_emission_prog_group, &hitgroup_records[sbt_idx]));
                Emission* emission_data = (Emission*)materials[meshID];
                hitgroup_records[sbt_idx].data.shading.emission.color = emission_data->color;
                break;
            }
            default:
                Throw("This material type is not supported\n");
            }
            
            hitgroup_records[sbt_idx].data.mesh.vertices = reinterpret_cast<float3*>(state.d_vertices[meshID]);
            hitgroup_records[sbt_idx].data.mesh.normals = reinterpret_cast<float3*>(state.d_normals[meshID]);
            hitgroup_records[sbt_idx].data.mesh.indices = reinterpret_cast<int3*>(state.d_indices[meshID]);
        }

        {
            const int sbt_idx = meshID * RAY_TYPE_COUNT + 1;
            memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
            switch(materials[meshID]->type()) {
            case MaterialType::Diffuse:
                OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_diffuse_prog_group, &hitgroup_records[sbt_idx]));
                break;
            case MaterialType::Dielectric:
                OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_dielectric_prog_group, &hitgroup_records[sbt_idx]));
                break;
            case MaterialType::Metal:
                OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_metal_prog_group, &hitgroup_records[sbt_idx]));
                break;
            case MaterialType::Emission:
                OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_emission_prog_group, &hitgroup_records[sbt_idx]));
                break;
            default:
                Throw("This material type is not supported\n");
            }
        }
    }

    CUdeviceptr d_hitgroup_records;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * meshes.size()
    ));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records.data(),
        hitgroup_record_size * RAY_TYPE_COUNT * meshes.size(),
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = hitgroup_records.size();
}

// ------------------------------------------------------------------------------------------------------------
// Cleanup state
// ------------------------------------------------------------------------------------------------------------
void cleanupState(PathTracerState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_dielectric_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_dielectric_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_diffuse_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_diffuse_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_emission_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_emission_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}

// ========== Main ==========
int main(int argc, char* argv[])
{
    PathTracerState state;
    state.params.width = 768;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    // Parse command line options
    std::string outfile;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--file" || arg == "-f")
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
            state.params.width = w;
            state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[i++]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        initTriangleMeshes();
        initCameraState();
        //createMaterials(state);
        
        // Set up OptiX state
        createContext(state);
        buildMeshAccel(state);
        createModule(state);
        createPipeline(state);
        createSBT(state);
        initLaunchParams(state);

        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixPathTracer", state.params.width, state.params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &state.params);

            // Render loop
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream(state.stream);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do 
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, state.params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, state);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    state.params.subframe_index++;
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
                state.params.width,
                state.params.height
            );

            handleCameraUpdate(state.params);
            handleResize(output_buffer, state.params);
            launchSubframe(output_buffer, state);

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

        cleanupState(state);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caugh exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}