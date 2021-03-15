#pragma once 

#include <core/scene.h>

namespace pt {

void Scene::prepare_on_device(const std::vector<ProgramGroup>& prg_groups, Params& params)
{
    params.width = m_width;
    params.height = m_height;
}

}