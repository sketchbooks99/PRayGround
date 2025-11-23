#include "envmap.h"
#include <prayground/core/cudabuffer.h>

namespace prayground {

    void EnvironmentEmitter::copyToDevice()
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        auto data = this->getData();

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data), 
            &data, sizeof(Data), 
            cudaMemcpyHostToDevice
        ));
    }

    EnvironmentEmitter::Data EnvironmentEmitter::getData() const 
    {
        return { m_texture->getData() };
    }

    // ---------------------------------------------------------------------------
    EnvmapSampleData createEnvmapSampleData(Vec4f* envmap_data, uint32_t width, uint32_t height)
    {
        float total_luminance = 0.0f;

        // Allocate buffers
        float* conditional_cdf = new float[width * height];
        float* row_luminance = new float[height];
        float* marginal_cdf = new float[height];

        // Build conditional CDF per row
        for (uint32_t y = 0; y < height; ++y) {
            float row_sum = 0.0f;
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t idx = y * width + x;

                float s = ((float)x + 0.5f) / width;
                float t = ((float)y + 0.5f) / height;

                // Compute luminance (using Rec. 709)
                Vec4f pixel = envmap_data[idx];
                float l = luminance(pixel);
                
                float theta = t * math::pi;
                float sin_theta = clamp(sinf(theta), 1e-6f, 1.0f);

                l *= sin_theta;

                row_sum += l;
                conditional_cdf[idx] = row_sum;
            }

            // Normalize conditional CDF
            if (row_sum > 0.0f) {
                for (uint32_t x = 0; x < width; x++) {
                    uint32_t idx = y * width + x;
                    conditional_cdf[idx] /= row_sum;
                }
            }

            row_luminance[y] = row_sum;
            total_luminance += row_sum;
        }

        // Build marginal CDF
        float marginal_sum = 0.0f;
        for (uint32_t y = 0; y < height; y++) {
            marginal_sum += row_luminance[y];
            marginal_cdf[y] = marginal_sum;
        }

        // Normalize marginal CDF
        if (marginal_sum > 0.0f) {
            for (uint32_t y = 0; y < height; y++) {
                marginal_cdf[y] /= marginal_sum;
            }
        }

        EnvmapSampleData sample_data;
        CUDABuffer<float> d_conditional_cdf;
        CUDABuffer<float> d_row_luminance;
        CUDABuffer<float> d_marginal_cdf;

        d_conditional_cdf.copyToDevice(conditional_cdf, sizeof(float) * width * height);
        d_row_luminance.copyToDevice(row_luminance, sizeof(float) * height);
        d_marginal_cdf.copyToDevice(marginal_cdf, sizeof(float) * height);

        sample_data.d_conditional_cdf = d_conditional_cdf.deviceData();
        sample_data.d_row_luminance = d_row_luminance.deviceData();
        sample_data.d_marginal_cdf = d_marginal_cdf.deviceData();
        sample_data.width = width;
        sample_data.height = height;
        sample_data.total_luminance = total_luminance;

        return sample_data;
    }
    
} // namespace prayground
