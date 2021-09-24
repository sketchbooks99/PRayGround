#include "camera.h"
#include <prayground/app/app_runner.h>
#include <prayground/math/util.h>

namespace prayground {

void Camera::enableTracking(std::shared_ptr<Window> window)
{
    window->events().mouseDragged.bindFunction([&](float x, float y, int button){ return this->mouseDragged(x, y, button); });
    window->events().mouseScrolled.bindFunction([&](float xoffset, float yoffset){ return this->mouseScrolled(xoffset, yoffset); });
}

void Camera::disableTracking()
{
    TODO_MESSAGE();
}

/// @todo FoxAxisへの対応
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