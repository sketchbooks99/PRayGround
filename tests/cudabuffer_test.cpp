#include <optix.h>
#include "../pathtracer/core/cudabuffer.h"

int main() {
    std::vector<float3> float3_array;
    for(int i = 0; i < 1000; i++) {
        float val = (float)i;
        float3_array.emplace_back(make_float3(val, val, val));
    }
    CUDABuffer<float3> float_buffer;
    size_t buffer_size = sizeof(float3) * float3_array.size();
    float_buffer.allocate(float3_array.data(), buffer_size);

}