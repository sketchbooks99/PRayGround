#pragma once

#include <prayground/core/texture.h>
#include <prayground/core/keypoint.h>
#ifndef __CUDACC__
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {
    template <typename T>
    class GradientTexture_ final : public Texture {
    public:
        using ColorType = T;

        struct Data {
            Keypoint<T>* keypoints;
            uint32_t num_keypoints;
            EaseType ease_type;
            Vec2f dir;
        };
#ifndef __CUDACC__
        GradientTexture_(std::vector<Keypoint<T>> keypoints, int prg_id, EaseType ease_type = EaseType::Linear, const Vec2f& dir = Vec2f(0, 1))
            : Texture(prg_id), m_keypoints(keypoints), m_dir(dir) {}

        constexpr TextureType type() override {
            return TextureType::Gradient;
        }

        void addKeypoint(const Keypoint<T>& keypoint) {
            m_keypoints.push_back(keypoint);
        }

        void removeKeypoint(int index) {
            if (index < 0 || index >= m_keypoints.size()) {
                PG_LOG_WARN("Invalid index: ", index);
                return;
            }
            m_keypoints.erase(m_keypoints.begin() + index);
        }

        void setEaseType(EaseType ease_type) {
            m_ease_type = ease_type;
        }

        void copyToDevice() override {
            if (m_keypoints.empty()) return;

            CUDABuffer<Keypoint<T>> d_keypoints;
            d_keypoints.copyToDevice(m_keypoints);
            Data data = { 
                .keypoints = d_keypoints.deviceData(), 
                .num_keypoints = static_cast<uint32_t>(m_keypoints.size()), 
                .ease_type = m_ease_type, 
                .dir = m_dir
            };

            if (!d_data) 
                CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
            CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
        }
    private:
        std::vector<Keypoint<T>> m_keypoints;
        EaseType m_ease_type;
        Vec2f m_dir;
#endif
    };
} // namespace prayground