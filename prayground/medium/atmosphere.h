#pragma once 

namespace prayground {

    template <typename T>
    class Atmosphere_ {
    public:
        struct Data {
            T sigma_a;
            T sigma_s;
            T sigma_t;
            float m_g;
        };

#ifndef __CUDACC__
        Atmosphere_(T sigma_a, T sigma_s, float g)
            : m_sigma_a(sigma_a), m_sigma_s(sigma_s), m_sigma_t(sigma_a + sigma_t), m_g(g) {}

        void copyToDevice()
        {
            Data data =
            {
                .sigma_a = m_sigma_a,
                .sigma_s = m_sigma_s,
                .sigma_t = m_sigma_t,
                .g = m_g,
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
        T m_sigma_a, m_sigma_s, m_sigma_t;
        float m_g;

        void* d_data;
#endif // __CUDACC__
    };

} // namespace prayground