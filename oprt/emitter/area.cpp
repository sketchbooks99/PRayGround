#include "area.h"
#include "../core/material.h"

namespace oprt {

AreaEmitter::AreaEmitter(
    const std::shared_ptr<Shape>& shape, 
    const std::shared_ptr<Texture>& texture,
    float intensity, 
    bool twosided
) 
: m_shape(shape)
, m_texture(texture)
, m_intensity(intensity)
, m_twosided(twosided) 
{

}

void AreaEmitter::prepareData() 
{
    m_texture->prepareData();

    
}

}