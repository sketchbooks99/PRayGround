#pragma once

#include <prayground/math/vec_math.h>

#ifndef __CUDACC__
    #include <prayground/app/window.h>
#endif

namespace prayground {

#ifndef __CUDACC__

/**
 * @brief 
 * Standard pinhole camera
 */
class Camera {
public:
    enum class FovAxis
    {
        Horizontal,
        Vertical,
        Diagonal
    };

    Camera() : 
        m_origin(make_float3(0.0f, 0.0f, -1.0f)), 
        m_lookat(make_float3(0.0f)),
        m_up(make_float3(0.0f, 1.0f, 0.0f)),
        m_fov(40.0f), 
        m_aspect(1.0f),
        m_nearclip(0.01f),
        m_farclip(10000.0f),
        m_fovaxis(FovAxis::Horizontal) {}

    Camera(const float3& origin, const float3& lookat, const float3& up, float fov, float aspect, 
        float nearclip = 0.01f, float farclip = 10000.0f, FovAxis fovaxis=FovAxis::Horizontal)
    : m_origin(origin), m_lookat(lookat), m_up(up), m_fov(fov), m_aspect(aspect) 
    , m_nearclip(nearclip), m_farclip(farclip), m_fovaxis(fovaxis) 
    {}

    float3 direction() const;

    const float3& origin() const;
    void setOrigin(const float3& origin);
    void setOrigin(float x, float y, float z);

    const float3& lookat() const;
    void setLookat(const float3& lookat);
    void setLookat(float x, float y, float z);

    const float3& up() const;
    void setUp(const float3& up);
    void setUp(float x, float y, float z);

    const float& fov() const;
    void setFov(float fov);

    const float& aspect() const;
    void setAspect( float aspect );

    const float& nearClip() const;
    void setNearClip( float nearclip );

    const float& farClip() const;
    void setFarClip( float farclip );

    const FovAxis& fovAxis() const;
    void setFovAxis( FovAxis fovaxis );

    void enableTracking(std::shared_ptr<Window> window);
    void disableTracking();

    void UVWFrame(float3& U, float3& V, float3& W) const;

protected:
    float3 m_origin;
    float3 m_lookat;
    float3 m_up;
    float m_fov;
    float m_aspect;
    float m_nearclip;
    float m_farclip;
    FovAxis m_fovaxis;
private:
    void mouseDragged(float x, float y, int button);
    void mouseScrolled(float xoffset, float yoffset);
};

/**
 * @brief 
 * Lens camera to realize a DoF (Depth of Field) effect.
 * This class implemented based on \c Camera class and extend basic features of it.
 */
class LensCamera final : public Camera {
public:
    LensCamera() : Camera(), m_aperture(0.01f), m_focal_length(100.0) {}
    LensCamera(
        const float3& origin, const float3& lookat, const float3& up, float fov, float aspect, 
        float nearclip = 0.01f, float farclip = 10000.0f, 
        float aperture = 0.01f, float focal_length = 100.0f,
        FovAxis fovaxis=FovAxis::Horizontal)
    : Camera(origin, lookat, up, fov, aspect, nearclip, farclip, fovaxis), m_aperture(aperture), m_focal_length(focal_length)
    {}

    const float& aperture() const { return m_aperture; }
    void setAperture( const float& aperture ) { m_aperture = aperture; }

    const float& focalLength() const { return m_focal_length; }
    void setFocalLength( const float& focal_length ) { m_focal_length = focal_length; }
private:
    float m_aperture;
    float m_focal_length;
};

#endif // __CUDACC__

}