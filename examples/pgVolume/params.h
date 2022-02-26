#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

struct LaunchParams {
	unsigned int width;
	unsigned int height;
	unsigned int samples_per_launch;
	unsigned int max_depth;
	int frame;
	uchar4* result;
	uchar4* accum;
	OptixTraversableHandle handle;
};