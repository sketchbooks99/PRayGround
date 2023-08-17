#include <prayground/optix/omm.h>
#include <prayground/texture/cuda/textures.cuh>

namespace prayground {

    extern "C" HOST void calculateOpacityMapOnDevice(
        unsigned short** out_opacity_data, 
        const Vec2f* texcoords, const Face* faces, 
        const void* texture_data, Texture::Type texture_type, 
        const OpacityMicroMap::Settings& settings)
    {

    }

} // namespace prayground