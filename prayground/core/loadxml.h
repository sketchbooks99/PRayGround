/**
 * @brief Xml parser for instant rendering via builtin scene class of @c pgScene.
 */ 

#pragma once 

#include <prayground/core/util.h>
#include <prayground/core/scene.h>
#include <prayground/emitter/envmap.h>
#include <prayground/emitter/area.h>
#include <vector>
#include <memory>
#include <filesystem>

namespace prayground {

void loadShape(const std::filesystem::path& filepath, 
    std::vector<std::shared_ptr<Shape>>& shapes, 
    std::vector<std::shared_ptr<Material>>& materials);

void loadTexture(const std::filesystem::path& filepath, std::vector<std::shared_ptr<Texture>>& textures);

void loadMaterial(const std::filesystem::path& filepath, std::vector<std::shared_ptr<Material>>& materials);

void loadEnvmap(const std::filesystem::path& filepath, EnvironmentEmitter& envmap);

void loadAreaEmitter(const std::filesystem::path& filepath, AreaEmitter& area);

void loadScene(const std::filesystem::path& filepath, pgScene& scene);

} // ::prayground