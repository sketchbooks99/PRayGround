#include <prayground/prayground.h>

extern "C" { __constant__ LaunchParams params; }

extern "C" __device__ void __raygen__thrust()
{
    int w = params.width;
    int h = params.height;
    Vec3ui idx(optixGetLaunchIndex());

    params.d_vector.push_back(int(idx.y() * w + idx.x()));
}

extern "C" __device__ void __miss__void()
{
    
}