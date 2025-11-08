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

#define SUBMISSION 0

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
    pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> buildTreeMeshWithAPI(uint32_t& seed);
    
    shared_ptr<TriangleMesh> buildVoronoiRockMesh();
    TerrainMeshResult buildTerrainMesh(TerrainParams params);

    Context m_ctx;
    CUstream m_stream;
    Pipeline m_ppl;

    LaunchParams m_params;

    Bitmap m_bitmap;
    FloatBitmap m_accum_buffer;

    static constexpr uint32_t NRay = 2;
    using AppScene = Scene<Camera, NRay>;
    AppScene m_scene;

    bool is_camera_updated;

    vector<shared_ptr<Curves>> m_tree_curves;
};