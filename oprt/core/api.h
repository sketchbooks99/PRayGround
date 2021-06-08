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



}