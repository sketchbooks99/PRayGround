#pragma once 

#include <prayground/core/shape.h>

#ifndef __CUDACC__
#include <memory>
#endif

namespace prayground {

    template <typename Spectrum>
    class GridMedium_ : public Shape {
    public:
        struct Data {
            Spectrum sigma_a;
            Spectrum sigma_s;
            float sigma_t;
            float g;
            int nx;
            int ny;
            int nz;
            float* density;
        };
#ifndef __CUDACC__
        GridMedium_(const Spectrum& sigma_a, const Spectrum& sigma_s, float g, 
                    int nx, int ny, int nz, const float* density)
            : m_sigma_a(sigma_a), m_sigma_s(sigma_s), 
              m_g(g), 
              m_nx(nx), m_ny(ny), m_nz(nz), 
              m_density(new float[nx * ny * nz])
        {
            m_sigma_t = (sigma_a + sigma_s)[0];
            memcpy(m_density.get(), density, sizeof(float) * nx * ny * nz);
        }

        constexpr ShapeType type() override 
        {
            return ShapeType::Custom;
        }

        void copyToDevice() override 
        {
            auto data = this->getData();

            // Copy data to device through Shape::d_data
            if (!d_data)
                CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
            CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
        }

        uint32_t numPrimitives() const override
        {
            return 1;
        }

        void free() override 
        {
            if (d_density) CUDA_CHECK(cudaFree(d_density));
            Shape::free();
        }

        AABB bound() const override 
        {
            return AABB {
                
            };
        }

        Data getData() 
        {
            // Copy density data to device
            const float sizeof_density = sizeof(float) * m_nx * m_ny * m_nz;
            if (!d_density)
                CUDA_CHECK(cudaMalloc(&d_density, sizeof_density));
            CUDA_CHECK(cudaMemcpy(d_density, m_density.get(), sizeof_density, cudaMemcpyHostToDevice));

            return Data {
                .sigma_a = m_sigma_a, 
                .sigma_s = m_sigma_s, 
                .sigma_t = m_sigma_t,
                .g = m_g, 
                .nx = m_nx, 
                .ny = m_ny, 
                .nz = m_nz,
                .density = d_density
            };
        }
    private:
        Spectrum m_sigma_a, m_sigma_s;
        float m_sigma_t;
        float m_g;
        int m_nx, m_ny, m_nz;
        std::unique_ptr<float[]> m_density;
        float* d_density;
#endif
    };

} // namespace prayground