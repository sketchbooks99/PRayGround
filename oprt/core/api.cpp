#include "api.h"

namespace oprt 
{

namespace {
    constexpr uint32_t num_builtin_materials = 4;
    constexpr uint32_t num_builtin_textures = 3;
    unsigned int current_callable_program_id = 0;
}

uint32_t oprtGetNumBuildInMaterials() { return num_buildin_materials; }
uint32_t oprtGetNumBuildInTextures()  { return num_buildin_textures; }

}