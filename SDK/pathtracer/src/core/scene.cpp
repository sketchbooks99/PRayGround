#pragma once 

#include <core/scene.h>

namespace pt {

void Scene::build_gas() {
    std::vector<OptixBuildInput> build_inputs_mesh;   // Mesh
    std::vector<OptixBuildInput> build_inputs_custom; // Custom primitives
}

void Scene::create_hitgroup_sbt(const OptixModule& module, OptixShaderBindingTable& sbt) {

}

}