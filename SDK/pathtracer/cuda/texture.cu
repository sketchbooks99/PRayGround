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

// Compute texture derivatives in texture space from texture derivatives in world space
// and  ray differentials.
inline __device__ void computeTextureDerivatives( float2&       dpdx,  // texture derivative in x (out)
                                                  float2&       dpdy,  // texture derivative in y (out)
                                                  const float3& dPds,  // world space texture derivative
                                                  const float3& dPdt,  // world space texture derivative
                                                  float3        rdx,   // ray differential in x
                                                  float3        rdy,   // ray differential in y
                                                  const float3& normal,
                                                  const float3& rayDir )
{
    // Compute scale factor to project differentials onto surface plane
    float s = dot( rayDir, normal );

    // Clamp s to keep ray differentials from blowing up at grazing angles. Prevents overblurring.
    const float sclamp = 0.1f;
    if( s >= 0.0f && s < sclamp )
        s = sclamp;
    if( s < 0.0f && s > -sclamp )
        s = -sclamp;

    // Project the ray differentials to the surface plane.
    float tx = dot( rdx, normal ) / s;
    float ty = dot( rdy, normal ) / s;
    rdx -= tx * rayDir;
    rdy -= ty * rayDir;

    // Compute the texture derivatives in texture space. These are calculated as the
    // dot products of the projected ray differentials with the texture derivatives. 
    dpdx = make_float2( dot( dPds, rdx ), dot( dPdt, rdx ) );
    dpdy = make_float2( dot( dPds, rdy ), dot( dPdt, rdy ) );
}

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


