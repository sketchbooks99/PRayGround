#include "envmap.h"
#include <oprt/core/file_util.h>
#include <oprt/core/material.h>
#include <oprt/texture/bitmap.h>

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
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    EnvironmentEmitterData data =
    {
        m_texture->devicePtr(),
        m_texture->programId()
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(EnvironmentEmitterData)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_data), 
        &data, sizeof(EnvironmentEmitterData), 
        cudaMemcpyHostToDevice
    ));
}

}
