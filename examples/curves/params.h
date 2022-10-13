#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

struct LightInfo
{
	Vec3f position;
	
	// Emittance for phong-shading
	Vec3f ambient;
	Vec3f diffuse;
	Vec3f specular;
};

struct PhongData
{
	Vec3f emission;
	Vec3f ambient;
	Vec3f diffuse;
	Vec3f specular;
	float shininess;
};

struct LaunchParams {
	uint32_t width;
	uint32_t height;
	uint32_t samples_per_launch;
	int32_t frame;

	LightInfo light;

	Vec4u* result_buffer;
	OptixTraversableHandle handle;
};