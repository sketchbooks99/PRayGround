#include "camera.h"
#include <prayground/app/app_runner.h>
#include <prayground/math/util.h>

namespace prayground {

// --------------------------------------------------------------------------------------
float3 Camera::direction() const 
{ 
    return normalize(m_lookat - m_origin); 
}

const float3& Camera::origin() const
{
    return m_origin;
}
void Camera::setOrigin(const float3& origin)
{
    m_origin = origin;
}
void Camera::setOrigin(float x, float y, float z)
{
    m_origin = make_float3(x, y, z);
}

const float3& Camera::lookat() const
{
    return m_lookat;
}
void Camera::setLookat(const float3& lookat)
{
    m_lookat = lookat;
}
void Camera::setLookat(float x, float y, float z)
{
    m_lookat = make_float3(x, y, z);
}

const float3& Camera::up() const
{
    return m_up;
}
void Camera::setUp(const float3& up)
{
    m_up = up;
}
void Camera::setUp(float x, float y, float z)
{
    m_up = make_float3(x, y, z);
}

const float& Camera::fov() const
{
    return m_fov;
}
void Camera::setFov(float fov)
{
    m_fov = fov;
}

const float& Camera::aspect() const
{
    return m_aspect;
}
void Camera::setAspect( float aspect )
{
    m_aspect = aspect;
}

const float& Camera::nearClip() const
{
    return m_nearclip;
}
void Camera::setNearClip( float nearclip )
{
    m_nearclip = nearclip;
}

const float& Camera::farClip() const
{
    return m_farclip;
}
void Camera::setFarClip( float farclip )
{
    m_farclip = farclip;
}

const Camera::FovAxis& Camera::fovAxis() const
{
    return m_fovaxis;
}
void Camera::setFovAxis( FovAxis fovaxis )
{
    m_fovaxis = fovaxis;
}

// --------------------------------------------------------------------------------------
void Camera::enableTracking(std::shared_ptr<Window> window)
{
    window->events().mouseDragged.bindFunction([&](float x, float y, int button){ return this->mouseDragged(x, y, button); });
    window->events().mouseScrolled.bindFunction([&](float xoffset, float yoffset){ return this->mouseScrolled(xoffset, yoffset); });
}

void Camera::disableTracking()
{
    UNIMPLEMENTED();
}

void Camera::UVWFrame(float3& U, float3& V, float3& W) const
{
    W = m_lookat - m_origin;
    float wlen = length(W);
    U = normalize(cross(W, m_up));
    V = normalize(cross(W, U));

    float vlen = wlen * tanf(0.5f * m_fov * math::pi / 180.0f);
    V *= vlen;
    float ulen = vlen * m_aspect;
    U *= ulen;
}

// --------------------------------------------------------------------------------------
void Camera::mouseDragged(float x, float y, int button)
{
    float deltaX = x - pgGetPreviousMousePosition().x;
    float deltaY = y - pgGetPreviousMousePosition().y;
    float cam_length = length(this->origin() - this->lookat());
    float3 cam_dir = normalize(this->origin() - this->lookat());

    float theta = acosf(cam_dir.y);
    float phi = atan2(cam_dir.z, cam_dir.x);

    theta = clamp(theta - math::radians(deltaY * 0.25f), math::eps, math::pi - math::eps);
    phi += math::radians(deltaX * 0.25f);

    float cam_x = cam_length * sinf(theta) * cosf(phi);
    float cam_y = cam_length * cosf(theta);
    float cam_z = cam_length * sinf(theta) * sinf(phi);

    this->setOrigin(this->lookat() + make_float3(cam_x, cam_y, cam_z));
}

void Camera::mouseScrolled(float xoffset, float yoffset)
{
    float zoom = yoffset < 0 ? 1.1f : 1.0f / 1.1f;
    this->setOrigin(this->lookat() + (this->origin() - this->lookat()) * zoom);
}

} // ::prayground