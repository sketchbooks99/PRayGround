#pragma once

#include "../oprt.h"

namespace oprt
{

void oprtEnvironment(const float3&);
// void oprtEnvironment(const std::shared_ptr<EnvironmentEmitter>& env);

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