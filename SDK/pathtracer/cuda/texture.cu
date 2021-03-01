#include <optix.h>

#include "core_util.h"

/**!
 Texture should be return evaluated color 
 at UV coordinates specified by intersection and/or closest-hit programs
  
 @param 
 * - coord
   - |float2|
   - UV coordinates to evaluate color of textures.
*/

/**! MEMO:
 There is no need to access SBT data through HitGroupData. 
 It is OK to connect programs and Texture by SBT. 
*/

// Constant texture
CALLABLE_FUNC float3 DC_FUNC(constant_eval) ( float2 coord ) {
    const Constant* constant = reinterpret_cast<Constant*>(optixGetSbtDataPointer());
    return constant->color;
}

// Checker texture
CALLABLE_FUNC float3 DC_FUNC(checker_eval) ( float2 coord ) {
    const Checker* checker = reinterpret_cast<Checker*>(optixGetSbtDataPointer());
    const float3 color1 = checker->color1;
    const float3 color2 = checker->color2;
    const float scale = checker->scale;

    float sines = sin(scale * coord.x) * sin(scale * coord.y);
    if (sines < 0)
        return color1;
    else 
        return color2;
}

CALLABLE_FUNC float3 DC_FUNC(image_eval) ( float2 coord ) {
    const Image* image = reinterpret_cast<Image*>(optixGetSbtDataPointer());
    
}


