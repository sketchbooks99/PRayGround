#include <prayground/prayground.h>

using namespace std;

int main() 
{
    SampledSpectrum::init(false);
    
    array<float3, 6> rgbs{ float3{0.2f, 0.5f, 1.0f}, float3{0.2f, 1.0f, 0.5f}, 
        float3{0.5f, 0.2f, 1.0f}, float3{1.0f, 0.2f, 0.5f}, 
        float3{1.0f, 0.5f, 0.2f}, float3{0.5f, 1.0f, 0.2f} };
    for (const float3& rgb : rgbs)
    {
        SampledSpectrum spd = SampledSpectrum::fromRGB(rgb);
        pgLog("[RGB] original:", rgb, "reconstructed:", spd.toRGB());
    }
    const float3 rgb{ 1.0f, 0.5f, 0.2f };
}