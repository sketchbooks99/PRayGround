#include "film.h"

namespace oprt {

template <typename PixelType>
Film<PixelType>::Film() {}

template <typename PixelType>
Film<PixelType>::~Film() {}

template <typename PixelType>
void Film<PixelType>::postProcess()
{
    /**
     * @todo
     * Apply the post-processing for the bitmap using GLSL shader.
     */
}

template <typaname PixelType>
void Film<PixelType>::setGamma(const float gamma)
{
    m_gamma = gamma;
}

template <typename PixelType>
float Film<PixelType>::gamma() const 
{
    return m_gamma;
}

template <typename PixelType>
void Film<PixelType>::setExposure(const float exposure)
{
    m_exposure = exposure; 
}

template <typename PixelType>
void Film<PixelType>::exposure() const 
{
    return m_exposure;
}

template class Film_<unsigned char> Film;
template class Film_<float> HdrFilm;

}