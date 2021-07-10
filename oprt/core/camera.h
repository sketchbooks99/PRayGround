#include "util.h"
#include <sutil/vec_math.h>

namespace oprt {

struct CameraData
{
    float3 origin;
    float3 lookat;
    float3 up;
    float aperture;
    float nearclip;
    float farclip;
};

/**
 * @brief 
 * Standard pinhole camera.
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

    float3 direction() const { return normalize(m_lookat - m_origin); }

    const float3& origin() const { return m_origin; }
    void setOrigin(const float3& origin) { m_origin = origin; }

    const float3& lookat() const { return m_lookat; }
    void setLookat(const float3& lookat) { m_lookat = lookat; }

    const float3& up() const { return m_up; }
    void setUp(const float3& up) { m_up = up; }

    const float& fov() const { return m_fov; }
    void setFov(const float& fov) { m_fov = fov; }

    const float& aspect() const { return m_aspect; }
    void setAspect( const float& aspect ) { m_aspect = aspect; }

    const float& nearClip() const { return m_nearclip; }
    void setNearClip( const float& nearclip ) { m_nearclip = nearclip; }

    const float& farClip() const { return m_farclip; }
    void setFarClip( const float& farclip ) { m_farclip = farclip; } 

    const FovAxis& fovAxis() const { return m_fovaxis; }
    void setFovAxis( FovAxis fovaxis ) { m_fovaxis = fovaxis; }

protected:
    float3 m_origin;
    float3 m_lookat;
    float3 m_up;
    float m_fov;
    float m_aspect;
    float m_nearclip;
    float m_farclip;
    FovAxis m_fovaxis;
};

/**
 * @brief 
 * Lens camera to realize a DoF (Depth of Field) effect.
 * This class implemented based on \c Camera class and extend basic features of it.
 */
class LensCamera final : public Camera {
public:
    LensCamera() : m_aperture(0.01f), m_focal_length(100.0f), Camera() {}
    LensCamera(
        const float3& origin, const float3& lookat, const float3& up, float fov, float aspect, 
        float nearclip = 0.01f, float farclip = 10000.0f, 
        float aperture = 0.01f, float focal_length = 100.0f,
        FovAxis fovaxis=FovAxis::Horizontal)
    : m_aperture(aperture), m_focal_length(focal_length), Camera(origin, lookat, up, fov, aspect, nearclip, farclip, fovaxis)
    {}

    /** @brief Aperture of lens */
    const float& aperture() const { return m_aperture; }
    void setAperture( const float& aperture ) { m_aperture = aperture; }

    /** @brief Focus length of lens */
    const float& focalLength() const { return m_focal_length; }
    void setFocalLength( const float& focal_length ) { m_focal_length = focal_length; }
private:
    float m_aperture;
    float m_focal_length;
};

}