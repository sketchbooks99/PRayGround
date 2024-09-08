#include "conductor.h"

namespace prayground {

    // ------------------------------------------------------------------
    Conductor::Conductor(
        const SurfaceCallableID& surface_callable_id, 
        const std::shared_ptr<Texture>& texture, 
        bool twosided, 
        Vec3f ior,
        float tf_thickness, 
        float tf_ior, 
        Vec3f extinction
    )
        : Material(surface_callable_id), m_texture(texture), m_twosided(twosided), m_ior(ior), m_tf_thickness(tf_thickness), m_tf_ior(tf_ior), m_extinction(extinction)
    {
    
    }

    Conductor::~Conductor()
    {

    }

    // ------------------------------------------------------------------
    SurfaceType Conductor::surfaceType() const
    {
        return SurfaceType::Reflection;
    }

    SurfaceInfo Conductor::surfaceInfo() const
    {
        ASSERT(d_data, "Material data on device hasn't been allocated yet.");

        return SurfaceInfo{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = SurfaceType::RoughReflection
        };
    }

    // ------------------------------------------------------------------
    void Conductor::copyToDevice()
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data), 
            cudaMemcpyHostToDevice
        ));
    }

    void Conductor::free()
    {
        m_texture->free();
        Material::free();
    }

    // ------------------------------------------------------------------
    void Conductor::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> Conductor::texture() const 
    {
        return m_texture;
    }

    void Conductor::setTwosided(bool twosided)
    {
        m_twosided = twosided;
    }

    bool Conductor::twosided() const
    {
        return m_twosided;
    }

    void Conductor::setIOR(const Vec3f& ior)
    {
        m_ior = ior;
    }

    Vec3f Conductor::ior() const
    {
        return m_ior;
    }

    void Conductor::setThinfilmThickness(float tf_thickness)
    {
        m_tf_thickness = tf_thickness;
    }
    float Conductor::thinfilmThickness() const 
    {
        return m_tf_thickness;
    }

    void Conductor::setThinfilmIOR(float tf_ior) 
    {
        m_tf_ior = tf_ior;
    }
    float Conductor::thinfilmIOR() const
    {
        return m_tf_ior;
    }

    void Conductor::setExtinction(const Vec3f& extinction)
    {
        m_extinction = extinction;
    }

    Vec3f Conductor::extinction() const
    {
        return m_extinction;
    }

    Conductor::Data Conductor::getData() const
    {
        return Data{
            .texture = m_texture->getData(),
            .twosided = m_twosided,
            .ior = m_ior,
            .tf_thickness = m_tf_thickness,
            .tf_ior = m_tf_ior,
            .extinction = m_extinction
        };
    }

} // namespace prayground