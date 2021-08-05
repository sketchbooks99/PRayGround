#pragma once

#include "../oprt.h"

namespace oprt
{

// APIs for core apps
void oprtFrameRate(const uint32_t framerate);
uint32_t oprtGetWidth();
uint32_t oprtGetHeight();
float2 oprtGetMousePosition();

// APIs for rendeirng
void oprtEnvironment(const float3&);
void oprtEnvironment(const std::filesystem::path& filepath);
void oprtEnvironment(const std::shared_ptr<EnvironmentEmitter>& env);
void oprtCreateSceneOnDevice(Scene scene, unsigned int device_id=0);

uint32_t oprtGetNumBuiltInMaterials();
uint32_t oprtGetNumBuiltInTextures();
 
OptixTraversableHandle oprtBuildGAS(const std::shared_ptr<Shape>& shape);
OptixTraversableHandle oprtBuildGAS(const std::vector<std::shared_ptr<Shape>>& shapes);
OptixTraversableHandle oprtBuildIAS();

ProgramGroup oprtCreateProgram(OptixProgramGroupKind kind, const Context& ctx, OptixProgramGroupDesc desc);
ProgramGroup oprtCreateRaygenProgram(const Context& ctx, const Module& module, const char* func_name);
ProgramGroup oprtCreateMissProgram(const Context& ctx, const Module& module, const char* func_name);
ProgramGroup oprtCreateCallableProgram(const Context& ctx, const Module& module, const char* dc_func_name, const char* cc_func_name);
ProgramGroup oprtCreateHitGroupProgram(const Context& ctx, const Module& module, const char* is_func_name, const char* ah_func_name, const char* ch_func_name);
ProgramGroup oprtCreateHitGroupProgram(const Context& ctx, const Module& module, const char* is_func_name, const char* ch_func_name);
ProgramGroup oprtCreateHitGroupProgram(const Context& ctx, const Module& module, const char* ch_func_name);

template <class T, class... Args> 
std::shared_ptr<T> oprtCreateMaterial(Args... args)
{
    static_assert(std::is_base_of_v<Material, T>, "The template class is not derived class of oprt::Material.");
    std::make_shared<T>(args...);
}

template <class T, class... Args> 
std::shared_ptr<T> oprtCreateShape(Args... args)
{
    static_assert(std::is_base_of_v<Shape, T>, "The template class is not derived class of oprt::Shape.");
    return std::make_shared<T>(args...);
}

template <class T, class... Args> 
std::shared_ptr<T> oprtCreateTexture(Args... args)
{
    static_assert(std::is_base_of_v<Texture, T>, "The template class is not derived class of oprt::Texture.");
    return std::make_shared<T>(args...);
}

}