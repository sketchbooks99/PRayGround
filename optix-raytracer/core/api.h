#pragma once

#include "scene.h"
#include "emitter.h"

namespace oprt {

void setEnvironment(const std::string& filename);
void setEnvironment(Texture* texture);

void render(const Scene& scene);

}