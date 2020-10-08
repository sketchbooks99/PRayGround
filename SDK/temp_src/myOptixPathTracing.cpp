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

#include "myOptixPathTracing.h"
#include "Model.h"
#include "CUDABuffer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip> 
#include <iostream> 
#include <sstream>
#include <string>

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state 
int32_t mouse_button = -1;
int32_t samples_per_launch = 16;

template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitGroupRecord;
typedef Record<BunnyGroupData> BunnyGroupRecord;


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};

struct Instance
{
    float transform[12];
};

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle gas_handle = 0; // Traversable handle for triangle AS
    CUdeviceptr d_gas_output_buffer = 0; // Triangle AS memory
    
    std::vector<CUdeviceptr> d_vertices;
    std::vector<CUdeviceptr> d_indices;
    std::vector<CUdeviceptr> d_normals;

    OptixModule ptx_module = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;

    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup occlusion_miss_group = 0;
    OptixProgramGroup radiance_hit_group = 0;
    OptixProgramGroup occlusion_hit_group = 0;

    CUstream stream = 0;
    Params params;
    Params* d_params;

    OptixShaderBindingTable sbt = {};
};

// scene data
const int32_t TRIANGLE_COUNT = 12;
const int32_t MAT_COUNT = 5;

// std::vector successfully worked as same as std::array;
std::vector<Vertex> g_vertices;
std::vector<Normal> g_normals;
std::vector<uint32_t> g_mat_indices;
std::vector<int3> g_faces;

// Model
TriangleMesh model("../../model/bunny.obj",                         // filename
            make_float3(556.0f/2.0f, 548.0f/2.0f, 559.2f/2.0f),     // center position
            150.0f,                                                 // size
            make_float3(1,1,-1),                                    // axis
            make_float3(0.05f, 0.90f, 0.90f),                       // diffuse color
            make_float3(0.0f, 0.0f, 0.0f));                         // emission color

std::vector<TriangleMesh> meshes(1, model);

TriangleMesh floor;
TriangleMesh ceiling;
TriangleMesh back_wall;
TriangleMesh right_wall;
TriangleMesh left_wall;
TriangleMesh ceiling_light;

void initCornel()
{
    std::vector<Vertex> floor_vertices;
    std::vector<Normal> floor_normals(6, Normal(0.0f, 1.0f, 0.0f, 0.0f));
    std::vector<int3> floor_indices;
    floor_vertices.emplace_back(  0.0f, 0.0f,   0.0f, 0.0f);
    floor_vertices.emplace_back(  0.0f, 0.0f, 559.2f, 0.0f);
    floor_vertices.emplace_back(556.0f, 0.0f, 559.2f, 0.0f);
    floor_vertices.emplace_back(  0.0f, 0.0f,   0.0f, 0.0f);
    floor_vertices.emplace_back(556.0f, 0.0f, 559.2f, 0.0f);
    floor_vertices.emplace_back(556.0f, 0.0f,   0.0f, 0.0f);
    floor_indices.emplace_back(make_int3(0, 1, 2));
    floor_indices.emplace_back(make_int3(3, 4, 5));
    floor = TriangleMesh(floor_vertices, floor_indices, floor_normals,
        make_float3(0.80f), make_float3(0.0f));
    meshes.emplace_back(floor);

    std::vector<Vertex> ceiling_vertices;
    std::vector<Normal> ceiling_normals(6, Normal(0.0f, -1.0f, 0.0f, 0.0f));
    std::vector<int3> ceiling_indices;
    ceiling_vertices.emplace_back(  0.0f, 548.8f,   0.0f, 0.0f);
    ceiling_vertices.emplace_back(556.0f, 548.8f,   0.0f, 0.0f);
    ceiling_vertices.emplace_back(556.0f, 548.8f, 559.2f, 0.0f);
    ceiling_vertices.emplace_back(  0.0f, 548.8f,   0.0f, 0.0f);
    ceiling_vertices.emplace_back(556.0f, 548.8f, 559.2f, 0.0f);
    ceiling_vertices.emplace_back(  0.0f, 548.8f, 559.2f, 0.0f);
    ceiling_indices.emplace_back(make_int3(0, 1, 2));
    ceiling_indices.emplace_back(make_int3(3, 4, 5));
    ceiling = TriangleMesh(ceiling_vertices, ceiling_indices, ceiling_normals,
        make_float3(0.80f), make_float3(0.0f));
    meshes.emplace_back(ceiling);

    std::vector<Vertex> back_wall_vertices;
    std::vector<Normal> back_wall_normals(6, Normal(0.0f, 0.0f, -1.0f, 0.0));
    std::vector<int3> back_wall_indices;
    back_wall_vertices.emplace_back(  0.0f,   0.0f, 559.2f, 0.0f);
    back_wall_vertices.emplace_back(  0.0f, 548.8f, 559.2f, 0.0f);
    back_wall_vertices.emplace_back(556.0f, 548.8f, 559.2f, 0.0f);
    back_wall_vertices.emplace_back(  0.0f,   0.0f, 559.2f, 0.0f);
    back_wall_vertices.emplace_back(556.0f, 548.8f, 559.2f, 0.0f);
    back_wall_vertices.emplace_back(556.0f,   0.0f, 559.2f, 0.0f);
    back_wall_indices.emplace_back(make_int3(0, 1, 2));
    back_wall_indices.emplace_back(make_int3(3, 4, 5));
    back_wall = TriangleMesh(back_wall_vertices, back_wall_indices, back_wall_normals,
        make_float3(0.80f), make_float3(0.0f));
    meshes.emplace_back(back_wall);

    std::vector<Vertex> right_wall_vertices;
    std::vector<Normal> right_wall_normals(6, Normal(1.0f, 0.0f, 0.0f, 0.0f));
    std::vector<int3> right_wall_indices;
    right_wall_vertices.emplace_back(0.0f,   0.0f,   0.0f, 0.0f);
    right_wall_vertices.emplace_back(0.0f, 548.8f,   0.0f, 0.0f);
    right_wall_vertices.emplace_back(0.0f, 548.8f, 559.2f, 0.0f);
    right_wall_vertices.emplace_back(0.0f,   0.0f,   0.0f, 0.0f);
    right_wall_vertices.emplace_back(0.0f, 548.8f, 559.2f, 0.0f);
    right_wall_vertices.emplace_back(0.0f,   0.0f, 559.2f, 0.0f);
    right_wall_indices.emplace_back(make_int3(0, 1, 2));
    right_wall_indices.emplace_back(make_int3(3, 4, 5));
    right_wall = TriangleMesh(right_wall_vertices, right_wall_indices, right_wall_normals,
        make_float3(0.05f, 0.80f, 0.05f), make_float3(0.0f));
    meshes.emplace_back(right_wall);

    std::vector<Vertex> left_wall_vertices;
    std::vector<Normal> left_wall_normals(6, Normal(-1.0f, 0.0f, 0.0f, 0.0f));
    std::vector<int3> left_wall_indices;
    left_wall_vertices.emplace_back(556.0f,   0.0f,   0.0f, 0.0f);
    left_wall_vertices.emplace_back(556.0f,   0.0f, 559.2f, 0.0f);
    left_wall_vertices.emplace_back(556.0f, 548.8f, 559.2f, 0.0f);
    left_wall_vertices.emplace_back(556.0f,   0.0f,   0.0f, 0.0f);
    left_wall_vertices.emplace_back(556.0f, 548.8f, 559.2f, 0.0f);
    left_wall_vertices.emplace_back(556.0f, 548.8f,   0.0f, 0.0f);
    left_wall_indices.emplace_back(make_int3(0, 1, 2));
    left_wall_indices.emplace_back(make_int3(3, 4, 5));
    left_wall = TriangleMesh(left_wall_vertices, left_wall_indices, left_wall_normals,
        make_float3(0.80f, 0.05f, 0.05f), make_float3(0.0f));
    meshes.emplace_back(left_wall);

    std::vector<Vertex> ceiling_light_vertices;
    std::vector<Normal> ceiling_light_normals(6, Normal(0.0f, -1.0f, 0.0f, 0.0f));
    std::vector<int3> ceiling_light_indices;
    ceiling_light_vertices.emplace_back(343.0f, 548.6f, 227.0f, 0.0f);
    ceiling_light_vertices.emplace_back(213.0f, 548.6f, 227.0f, 0.0f);
    ceiling_light_vertices.emplace_back(213.0f, 548.6f, 332.0f, 0.0f);
    ceiling_light_vertices.emplace_back(343.0f, 548.6f, 227.0f, 0.0f);
    ceiling_light_vertices.emplace_back(213.0f, 548.6f, 332.0f, 0.0f);
    ceiling_light_vertices.emplace_back(343.0f, 548.6f, 332.0f, 0.0f);
    ceiling_light_indices.emplace_back(make_int3(0, 1, 2));
    ceiling_light_indices.emplace_back(make_int3(3, 4, 5));
    ceiling_light = TriangleMesh(ceiling_light_vertices, ceiling_light_indices, ceiling_light_normals,
        make_float3(0.0f), make_float3(15.0f));
    meshes.emplace_back(ceiling_light);
}

void initVertices() {
    // Floor -- white lambert
    g_vertices.emplace_back(   0.0f,   0.0f,   0.0f, 0.0f );
    g_vertices.emplace_back(   0.0f,   0.0f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f,   0.0f, 559.2f, 0.0f );

    g_vertices.emplace_back(   0.0f, 0.0f,   0.0f, 0.0f );
    g_vertices.emplace_back( 556.0f, 0.0f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f, 0.0f,   0.0f, 0.0f );

    // Ceiling -- white lambert
    g_vertices.emplace_back(   0.0f, 548.8f,   0.0f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f,   0.0f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f, 559.2f, 0.0f );

    g_vertices.emplace_back(   0.0f, 548.8f,   0.0f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f, 559.2f, 0.0f );
    g_vertices.emplace_back(   0.0f, 548.8f, 559.2f, 0.0f );

    // Back wall -- white lambert
    g_vertices.emplace_back(   0.0f,   0.0f, 559.2f, 0.0f );
    g_vertices.emplace_back(   0.0f, 548.8f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f, 559.2f, 0.0f );

    g_vertices.emplace_back(   0.0f,   0.0f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f,   0.0f, 559.2f, 0.0f );

    // Right wall -- green lambert
    g_vertices.emplace_back( 0.0f,   0.0f,   0.0f, 0.0f );
    g_vertices.emplace_back( 0.0f, 548.8f,   0.0f, 0.0f );
    g_vertices.emplace_back( 0.0f, 548.8f, 559.2f, 0.0f );

    g_vertices.emplace_back( 0.0f,   0.0f,   0.0f, 0.0f );
    g_vertices.emplace_back( 0.0f, 548.8f, 559.2f, 0.0f );
    g_vertices.emplace_back( 0.0f,   0.0f, 559.2f, 0.0f );

    // Left wall -- red lambert
    g_vertices.emplace_back( 556.0f,   0.0f,   0.0f, 0.0f );
    g_vertices.emplace_back( 556.0f,   0.0f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f, 559.2f, 0.0f );

    g_vertices.emplace_back( 556.0f,   0.0f,   0.0f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f, 559.2f, 0.0f );
    g_vertices.emplace_back( 556.0f, 548.8f,   0.0f, 0.0f );

    // Ceiling light -- emissive
    g_vertices.emplace_back( 343.0f, 548.6f, 227.0f, 0.0f );
    g_vertices.emplace_back( 213.0f, 548.6f, 227.0f, 0.0f );
    g_vertices.emplace_back( 213.0f, 548.6f, 332.0f, 0.0f );

    g_vertices.emplace_back( 343.0f, 548.6f, 227.0f, 0.0f );
    g_vertices.emplace_back( 213.0f, 548.6f, 332.0f, 0.0f );
    g_vertices.emplace_back( 343.0f, 548.6f, 332.0f, 0.0f );

    g_vertices.insert(g_vertices.end(), model.vertices.begin(), model.vertices.end());
}

void initNormals()
{
    // Floor
    g_normals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);

    // Ceiling
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);

    // Back wall
    g_normals.emplace_back(0.0f, 0.0f, -1.0f, 0.0f);
    g_normals.emplace_back(0.0f, 0.0f, -1.0f, 0.0f);
    g_normals.emplace_back(0.0f, 0.0f, -1.0f, 0.0f);
    g_normals.emplace_back(0.0f, 0.0f, -1.0f, 0.0f);
    g_normals.emplace_back(0.0f, 0.0f, -1.0f, 0.0f);
    g_normals.emplace_back(0.0f, 0.0f, -1.0f, 0.0f);

    // Right wall 
    g_normals.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);

    // Left wall
    g_normals.emplace_back(-1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(-1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(-1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(-1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(-1.0f, 0.0f, 0.0f, 0.0f);
    g_normals.emplace_back(-1.0f, 0.0f, 0.0f, 0.0f);

    // Ceiling light
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);
    g_normals.emplace_back(0.0f, -1.0f, 0.0f, 0.0f);

    // Push back model normals
    g_normals.insert(g_normals.end(), model.normals.begin(), model.normals.end());
}

void initMatIndices()
{
    g_mat_indices.emplace_back(0); g_mat_indices.emplace_back(0); // Floor          -- white lambert
    g_mat_indices.emplace_back(0); g_mat_indices.emplace_back(0); // Ceiling        -- white lambert
    g_mat_indices.emplace_back(0); g_mat_indices.emplace_back(0); // Back wall      -- white lambert
    g_mat_indices.emplace_back(1); g_mat_indices.emplace_back(1); // Right wall     -- green lambert
    g_mat_indices.emplace_back(2); g_mat_indices.emplace_back(2); // Left wall      -- red lambert
    g_mat_indices.emplace_back(3); g_mat_indices.emplace_back(3); // Ceiling light  -- emissive

    std::vector<uint32_t> model_indices(model.faces.size(), static_cast<uint32_t>(4));
    g_mat_indices.insert(g_mat_indices.end(), model_indices.begin(), model_indices.end());
}

void initFaces() {
    // 12 = num of triangle faces in cornel box.
    for (int i = 0; i < 12; i++)
    {
        g_faces.emplace_back(make_int3(i * 3, i * 3 + 1, i * 3 + 2));
    }

    // Add face index of the last triangle in cornel box to model.faces
    for (auto& face : model.faces) {
        auto last_idx = g_faces.back().z + 1;
        face += make_int3(last_idx, last_idx, last_idx);
    }

    g_faces.insert(g_faces.end(), model.faces.begin(), model.faces.end());
}

const std::array<float3, MAT_COUNT> g_emission_colors = 
{ {
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {15.0f, 15.0f, 15.0f},
    {0.0f, 0.0f, 0.0f} // for debugging "bunny.obj"
} };

const std::array<float3, MAT_COUNT> g_diffuse_colors = 
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f },
    { 0.05f, 0.80f, 0.80f } // for debugging "bunny.obj"
} };

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

    state.params.light.emission = make_float3( 15.0f, 15.0f, 5.0f );
    state.params.light.corner   = make_float3( 343.0f, 548.5f, 227.0f );
    state.params.light.v1       = make_float3( 0.0f, 0.0f, 105.0f );
    state.params.light.v2       = make_float3( -130.0f, 0.0f, 0.0f );
    state.params.light.normal   = normalize( cross( state.params.light.v1, state.params.light.v2 ) );
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

void buildMeshAccel( PathTracerState& state )
{
    // Initialize mesh data.
    /*initVertices();
    initMatIndices();
    initNormals();
    initFaces();*/
    initCornel();

    std::cerr << "Num vertices: " << g_vertices.size() << std::endl;
    std::cerr << "Mat indices: " << g_mat_indices.size() << std::endl;
    std::cerr << "Num faces: " << g_faces.size() << std::endl;

    // ref: https://github.com/ingowald/optix7course/blob/master/example08_addingTextures/SampleRenderer.cpp

    for (int meshID = 0; meshID < meshes.size(); meshID++)
    {

    }
    
    // alloc and copy vertices data
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof(Vertex);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
               reinterpret_cast<void*>(state.d_vertices),
               g_vertices.data(), vertices_size_in_bytes,
               cudaMemcpyHostToDevice
               ));

    // alloc and copy indices data
    const size_t indices_size_in_bytes = g_faces.size() * sizeof(int3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_indices), indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_indices),
        g_faces.data(), indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    // alloc and copy normals data
    if (!g_normals.empty())
    {
        const size_t normals_size_in_bytes = g_normals.size() * sizeof(Vertex);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_normals), normals_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.d_normals),
            g_normals.data(), normals_size_in_bytes,
            cudaMemcpyHostToDevice
        ));
    }

    // alloc and copy material data
    CUdeviceptr d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        g_mat_indices.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    uint32_t triangle_input_flags[MAT_COUNT] =
    {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    std::vector<OptixBuildInput> triangle_inputs;

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(g_vertices.size());
    triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(int3);
    triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(g_faces.size());
    triangle_input.triangleArray.indexBuffer = state.d_indices;
    triangle_input.triangleArray.numSbtRecords = MAT_COUNT;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    // ========== Memo : Custom primitive ===========
    // For creating build input of custom primitive, 
    // build input configuration will be as follows.
    //
    // OptixBuildInputCustomPrimitiveArray& build_input = buildInputs[0].customPrimitiveArray;
    // build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    // build_input.aabbBuffers = d_aabbBuffer;
    // build_input.numPrimitives = numPrimitives;
    // 
    // ==============================================


    // ========== Memo : Instance build input ==========
    // Unlike the triangle and AABB inputs, optixAccelBuild only accepts a single instance
    // build input per build call. Theare are upper limits to the possible number of 
    // instances (the size of the buffer of the OptixInstance structs), the SBT offset,
    // the visiblity mask, as well as the user ID.
    // 
    // An example of this sequence is as follows...
    // 
    // OptixInstance instance;
    // float transform[12] = {1,0,0,3,0,1,0,0,0,0,1,0};
    // memcpy(instance.transform, transform, sizeof(float)*12);
    // instance.instanceId = 0;
    // instance.visibilityMask = 255;
    // instance.sbtOffset = 0;
    // instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    // instance.traversableHandle = gasTraversable;
    //
    // void* d_instance;
    //
    // cudaMalloc(&d_instance, sizeof(OptixInstance));
    // cudaMemcpy(d_instance, &instance,
    //            sizeof(OptixInstance),
    //            cudaMemcpyHostToDevice);
    // 
    // OptixBuildInputInstanceArray* build_input = &buildInputs[0].instanceArray;
    // build_input->type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    // build_input->instances = d_instance;
    // build_input->numInstances = 1;
    // 
    // ==================================================
    
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // ========== Memo : Dynamic updates of AS ==========
    // To allow updates of an acceleration structure, 
    // set OPTIX_BUILD_FLAG_ALLOW_UPDATE in the build flags.
    //
    // For example:
    // 
    // accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    // accel_options.operation = OPTIX_BUILD_OPERATION_BUILD
    // 
    // To update the previously build acceleration structure, 
    // set OPTIX_BUILD_OPERATION_UPDATE and the call optix AccelBuild
    // on the same output data. All other options are required to be
    // identical to the original build. The update is done in-place
    // on the output data.
    //
    // For example:
    // 
    // accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    //
    // void* d_temp_update;
    // cudaMalloc(&d_temp_update, bufferSizes.tempUpdateSizeInBytes);
    //
    // optixAccelBuild(optixContext, cuStream, &accel_options,
    //      build_inputs, 2, d_temp_update,
    //      bufferSizes.tempUpdateSizeInBytes, d_output,
    //      bufferSizes.outputSizeInBytes, &outputHandle, nullptr, 0);
    // 
    // ==================================================

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &triangle_input, 
        1, 
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
        &triangle_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty,
        1
    ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_mat_indices ) ) );

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

void createModule( PathTracerState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 2;
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
        const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "myOptixPathTracing.cu");
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

    // TODO: If I would like to use multiple .cu source, Must I create other PTX module? 
    // - Absolutely YES, haha :)
    /*{
        const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "bsdf.cu");
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
    }*/

}

void createProgramGroups( PathTracerState& state )
{
    OptixProgramGroupOptions program_group_options = {};

    char log[2048];
    size_t sizeof_log = sizeof( log );

    {
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
            &state.raygen_prog_group
        ));
    }

    {
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
            &state.radiance_miss_group
        ));

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
            &state.occlusion_miss_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &state.radiance_hit_group
        ));

        memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &state.occlusion_hit_group
        ));
    }
}

void createPipeline(PathTracerState& state)
{
    // May be this is pipeline configration.
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.occlusion_miss_group,
        state.radiance_hit_group,
        state.occlusion_hit_group
    };

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
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]), // All groups have same size?
        log,
        &sizeof_log,
        &state.pipeline
    ));

    // We need to specify the max traversal depth. 
    // Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_miss_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_miss_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_hit_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_hit_group, &stack_sizes));

    uint32_t max_trace_depth = 2;
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

// SBT: Shader Binding Table
void createSBT(PathTracerState& state) 
{
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_miss_group, &ms_sbt[1]));
    ms_sbt[1].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT
    ));

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; i++)
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0;

            OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.emission_color = g_emission_colors[i];
            hitgroup_records[sbt_idx].data.diffuse_color = g_diffuse_colors[i];
            hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(state.d_vertices);
            hitgroup_records[sbt_idx].data.normals = reinterpret_cast<float4*>(state.d_normals);
            hitgroup_records[sbt_idx].data.indices = reinterpret_cast<int3*>(state.d_indices);
        }

        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;
            memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
            OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_hit_group, &hitgroup_records[sbt_idx]));
        }
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records,
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT,
        cudaMemcpyHostToDevice
    ));

    // ========== For bunny object ==========
    CUdeviceptr d_bunnygroup_records;
    const size_t bunnygroup_record_size = sizeof(BunnyGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_bunnygroup_records),
        bunnygroup_record_size * RAY_TYPE_COUNT
    ));

    BunnyGroupRecord bunnygroup_records[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_bunny_hit_group, &bunnygroup_records[0]));
    bunnygroup_records[0].data.diffuse_color = g_diffuse_colors[4];
    bunnygroup_records[0].data.vertices = reinterpret_cast<float4*>(state.d_vertices);
    bunnygroup_records[0].data.normals = reinterpret_cast<float4*>(state.d_normals);
    bunnygroup_records[0].data.indices = reinterpret_cast<int3*>(state.d_indices);

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT * MAT_COUNT;
}

void cleanupState(PathTracerState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    /*CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_vertices.data())));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_indices.data())));*/
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
        initCameraState();

        // Set up OptiX state
        createContext(state);
        buildMeshAccel(state);
        createModule(state);
        createProgramGroups(state);
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