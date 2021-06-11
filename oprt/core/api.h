#pragma once

#include "../oprt.h"

namespace oprt
{

void oprtEnvironment(const float3&);
// void oprtEnvironment(const std::shared_ptr<EnvironmentEmitter>& env);

void oprtCreateRaygenProgram();
void oprtCreateMissProgram();
void oprtCreateCallableProgram(const OptixDeviceContext& ctx, const Module& module);
void oprtCreateHitGroupProgram();

/**
 * @todo Implement adding custom material by calling this function as like follows
 * auto material = oprtCreateCustomMaterial<CustomMaterial>(args...)
 */
template <class T, class... Args> 
std::shared_ptr<T> oprtCreateCustomMaterial(Args... args)
{

}

template <class T, class... Args> 
std::shared_ptr<T> oprtCreateCustomShape(Args... args)
{

}

template <class T, class... Args> 
std::shared_ptr<T> oprtCreateCustomTexture(Args... args)
{

}

template <class... Args>
auto oprtCreateDefinedMaterial(const std::string& name, Args... args)
{
    
}

template <class... Args> 
auto oprtCreateDefinedShape(const std::string& name, Args... args) 
{

}

template <class... Args> 
auto oprtCreateDefinedTexture(const std::string& name, Args... args)
{

}




}