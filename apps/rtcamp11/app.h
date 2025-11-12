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

#include "svgf.cuh"

using namespace std;

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
    void resetMovie();
    void copyAreaEmitterToDevice();
    
    // New Tree API version (for testing new implementation)
    pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> buildTreeMesh(
        uint32_t& seed, Tree tree, TreeParam params, const vector<int>& leaf_texture_ids);
    
    // Recursive branch mesh building helper
    int buildBranchMeshRecursive(
        const Stem& stem,
        vector<Vec3f>& vertices,
        vector<Vec3f>& normals,
        vector<Vec2f>& texcoords,
        vector<Face>& faces,
        int radial_segments,
        int parent_ring_start = -1,
        float parent_t_offset = 0.0f,  // UV offset from parent for continuity
        const Vec3f* parent_right = nullptr,   // Frame from parent to avoid twisting
        const Vec3f* parent_tangent = nullptr);
    
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

    FloatBitmap m_bloom_bitmap;

    // Bloom effect buffers (Vec4f version)
    Vec4f* d_bloom_temp1 = nullptr;
    Vec4f* d_bloom_temp2 = nullptr;
    Vec4f* d_firefly_temp = nullptr;
    bool enable_bloom = true;
    float bloom_threshold = 1.0f;
    float bloom_intensity = 0.3f;
    int bloom_radius = 5.0f;
    float bloom_sigma = 5.0f;

    bool enable_firefly_filter = true;
    bool enable_gbuffer = false;

    FloatBitmap m_albedo_bitmap, m_normal_bitmap, m_uv_bitmap;
    static constexpr uint32_t NRay = 2;
    using AppScene = Scene<Camera, NRay>;
    AppScene m_scene;

    bool is_camera_updated;

    vector<shared_ptr<Curves>> m_tree_curves;

    vector<KeyPoint<Vec3f>> m_cam_points;
    vector<KeyPoint<Vec3f>> m_look_points;
    vector<KeyPoint<Vec3f>> m_light_points;

    map<string, AreaEmitterInfo> m_light_infos;
    CUDABuffer<AreaEmitterInfo> d_light_infos;

    float max_spp_for_debug = 128;

    Vec3f m_bunny1_pos;
    float m_bunny1_scale;

    Vec3f m_bunny2_pos;
    float m_bunny2_scale;
    
    Vec3f m_bunny3_pos;
    float m_bunny3_scale;

    // For debugging
    float m_frame_time;
    float m_interval;
    EaseType m_camera_ease;
    EaseType m_light_ease;
    int n_frame;
#if !INTERACTIVE && !SUBMISSION
    static constexpr uint32_t SPP = 16;
#else
    static constexpr uint32_t SPP = 128;
#endif
    static constexpr uint32_t SPP_PER_LAUNCH = 1;
    static constexpr uint32_t NUM_ITER = SPP / SPP_PER_LAUNCH;
    static constexpr float FPS = 12.0f;
    static constexpr float VIDEO_LENGTH = 6.0f;

    static constexpr bool ADAPTIVE_SAMPLING = true;
    static constexpr uint32_t ADAPTIVE_MIN_SAMPLES = 40;
};