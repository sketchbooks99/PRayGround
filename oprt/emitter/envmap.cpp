#include "envmap.h"
#include "../core/file_util.h"
#include "../core/material.h"
#include "../texture/bitmap.h"

namespace oprt {

EnvironmentEmitter::EnvironmentEmitter(const std::filesystem::path& filename)
{
    auto ext = getExtension(filename);
    if (ext == ".png" || ext == ".PNG" || ext == ".jpg" || ext == ".JPG")
        m_texture = std::make_shared<BitmapTexture>(filename);
    else if (ext == ".exr" || ext == ".EXR")
        m_texture = std::make_shared<BitmapTextureFloat>(filename);
}

void EnvironmentEmitter::prepareData()
{
    m_texture->prepareData();

    EnvironmentEmitterData data = 
    {
        m_texture->devicePtr(), 
        static_cast<uint8_t>(m_texture->type()) + static_cast<uint8_t>(MaterialType::Count) * RAY_TYPE_COUNT
    };

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(EnvironmentEmitterData)));
    CUDA_CHECK(cudaMemcpy(
        d_data, 
        &data, sizeof(EnvironmentEmitterData), 
        cudaMemcpyHostToDevice
    ));
}

}
