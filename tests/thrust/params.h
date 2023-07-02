#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace prayground;

struct LaunchParams {
    int width, height;

    thrust::device_vector<int> d_vector;
};