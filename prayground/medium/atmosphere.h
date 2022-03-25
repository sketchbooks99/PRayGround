#pragma once 

namespace prayground {

    template <typename Spectrum>
    class Atmosphere_ {
    public:
        struct Data {
            Spectrum sigma_a;
            Spectrum sigma_s;
            Spectrum sigma_t;
            float m_g;
        };

#ifndef __CUDACC__
        Atmosphere_(const Spectrum& sigma_a, const Spectrum& sigma_s, float g)
            : m_sigma_a(sigma_a), m_sigma_s(sigma_s), m_sigma_t(sigma_a + sigma_s), m_g(g) {}

        void copyToDevice()
        {
            Data data =
            {
                .sigma_a = m_sigma_a,
                .sigma_s = m_sigma_s,
                .sigma_t = m_sigma_t,
                .g = m_g
            };

            if (!d_data)
                CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
            CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
        }

        void free()
        {
            if (d_data) CUDA_CHECK(cudaFree(d_data));
            d_data = nullptr;
        }

    private:
        Spectrum m_sigma_a, m_sigma_s, m_sigma_t;
        float m_g;

        // Device side pointer
        void* d_data;
#endif // __CUDACC__
    };

} // namespace prayground