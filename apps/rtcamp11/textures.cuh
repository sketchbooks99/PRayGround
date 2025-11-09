#pragma once

#include <prayground/math/vec.h>
#include <prayground/math/noise.h>
#include <cuda_runtime.h>

namespace prayground {

    // Device-side Perlin Noise data (no dynamic allocation)
    struct PerlinNoiseData {
        static const int POINT_COUNT = 256;
        Vec3f rnd_vec[POINT_COUNT];
        int perm_x[POINT_COUNT];
        int perm_y[POINT_COUNT];
        int perm_z[POINT_COUNT];
        
        __device__ float noise(const Vec3f& p) const {
            float u = p.x() - floor(p.x());
            float v = p.y() - floor(p.y());
            float w = p.z() - floor(p.z());

            int i = static_cast<int>(floor(p.x()));
            int j = static_cast<int>(floor(p.y()));
            int k = static_cast<int>(floor(p.z()));
            Vec3f c[2][2][2];

            for(int di=0; di<2; di++) {
                for(int dj=0; dj<2; dj++) {
                    for(int dk=0; dk<2; dk++) {
                        c[di][dj][dk] = rnd_vec[
                            perm_x[(i+di) & 255] ^ 
                            perm_y[(j+dj) & 255] ^ 
                            perm_z[(k+dk) & 255]
                        ];
                    }
                }
            }
            return perlinInterop(c, u, v, w);
        }
        
        __device__ float turb(const Vec3f& p, int depth=7) const {
            float accum = 0.0f;
            Vec3f tmp_p = p;
            float weight = 1.0f;

            for (int i = 0; i < depth; i++) {
                accum += weight * noise(tmp_p);
                weight *= 0.5f;
                tmp_p *= 2.0f;
            }
            return fabs(accum);
        }
    };

    struct RockTexture {
        struct {
            int depth;
            float scale;
            float amplitude;
        } noise1, noise2;
    };
    
    // Tree Bark Texture Types
    enum class BarkType {
        ROUGH,      // Worley crackle - ごつごつした樹皮
        AGED,       // Layered FBM - 年季入った樹皮
        SMOOTH      // Flow lines - 滑らかな流線型
    };
    
    struct TreeBarkTexture {
        // Common parameters
        float bump_strength;            // Overall displacement strength
        uint32_t seed;
        BarkType type;
        
        // Rough Bark (Worley-based)
        struct {
            float cell_scale;           // Worley cell size
            float vertical_stretch;     // Y方向の引き伸ばし (2.0-3.0)
            float crack_depth;          // 亀裂の深さ
            float crack_threshold;      // F2-F1 threshold
        } rough;
        
        // Aged Bark (FBM-based)
        struct {
            int octaves;                // FBM octaves
            float scale;                // Base scale
            float warp_strength;        // Domain warping strength
            float vertical_bias;        // 縦方向のバイアス
        } aged;
        
        // Smooth Bark (Flow-based)
        struct {
            float flow_scale;           // Flow line scale
            float flow_strength;        // Flow direction strength
            float ripple_frequency;     // 横方向の波紋
            float smoothness;           // 滑らかさ (0-1)
        } smooth;
    };

    extern "C" float* bakeRockTexture(
        uint32_t seed,
        RockTexture rock_data,
        uint32_t width,
        uint32_t height
    );
    
    extern "C" float* bakeTreeBarkTexture(
        TreeBarkTexture bark_data,
        uint32_t width,
        uint32_t height
    );
    
    extern "C" Vec4f* bakeSmoothBarkColorTexture(
        TreeBarkTexture bark_data,
        uint32_t width,
        uint32_t height
    );

    extern "C" Vec4f* createBumpTextureFromHeightmap(
        float* d_heightmap,
        uint32_t width,
        uint32_t height,
        float bump_strength
    );
} // namespace prayground