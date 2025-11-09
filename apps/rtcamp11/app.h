#pragma once

#include <prayground/prayground.h>
#include "params.h"

#ifndef __CUDACC__
#include <prayground/physics/cuda/tree.cuh>
#include <prayground/physics/cuda/voronoi_rock.cuh>
#include <prayground/physics/terrain.h>
#include <prayground/physics/tree.h>
#endif

#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

using namespace std;

#define SUBMISSION 1

class App : public BaseApp 
{
public:
    void setup();
    void update();
    void draw();

    void mousePressed(float x, float y, int button);
    void mouseDragged(float x, float y, int button);
    void mouseReleased(float x, float y, int button);
    void mouseMoved(float x, float y);
    void mouseScrolled(float xoffset, float yoffset);

    void keyPressed(int key);
    void keyReleased(int key);
private:
    struct TerrainMeshResult {
        shared_ptr<TriangleMesh> mesh;
        shared_ptr<FloatBitmap> heightmap;
        float min_height;
        float max_height;
    };
    
    void initResultBufferOnDevice();
    void handleCameraUpdate();
    //shared_ptr<TriangleMesh> buildTreeMesh(ProceduralTreeData tree);
    pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> buildTreeMesh(ProceduralTreeData tree, uint32_t& seed, vector<shared_ptr<BitmapTexture>> leaf_textures);
    
    // New Tree API version (for testing new implementation)
    pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> buildTreeMeshWithAPI(uint32_t& seed, int n_leaf_textures = 1);
    
    shared_ptr<TriangleMesh> buildVoronoiRockMesh(const VoronoiRockParams& params);
    TerrainMeshResult buildTerrainMesh(TerrainParams params);

    Context m_ctx;
    CUstream m_stream;
    Pipeline m_ppl;

    LaunchParams m_params;

    Bitmap m_bitmap;
    FloatBitmap m_accum_buffer;

    // Float bitmap for bloom/post-processing (also used by denoiser)
    FloatBitmap m_float_bitmap;

    // Bloom effect buffers (Vec4f version)
    Vec4f* d_bloom_temp1 = nullptr;
    Vec4f* d_bloom_temp2 = nullptr;
    bool enable_bloom = true;
    float bloom_threshold = 1.0f;
    float bloom_intensity = 0.3f;
    int bloom_radius = 10;
    float bloom_sigma = 5.0f;

#if DENOISE
    Denoiser m_denoiser;
    Denoiser::Data m_denoise_data;
    FloatBitmap m_accum_bitmap, m_albedo_bitmap, m_normal_bitmap;
#endif

    static constexpr uint32_t NRay = 2;
    using AppScene = Scene<Camera, NRay>;
    AppScene m_scene;

    bool is_camera_updated;

    vector<shared_ptr<Curves>> m_tree_curves;
};