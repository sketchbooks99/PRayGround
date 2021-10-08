#pragma once

#include <vector>
#include <variant>

#include <prayground/core/shape.h>
#include <prayground/core/material.h>
#include <prayground/core/texture.h>
#include <prayground/core/camera.h>

#include <prayground/optix/sbt.h>
#include <prayground/optix/pipeline.h>

#include <prayground/emitter/envmap.h>
#include <prayground/emitter/area.h>

namespace prayground {

class pgScene {
public:
    void createHitGroupSBT();

    void addShape()
private:
    using pgRaygenRecord = Record<pgRaygenData>;
    using pgHitgroupRecord = Record<pgHitgroupData>;
    using pgMissRecord = Record<pgMissData>;
    using pgCallableRecord = Record<pgCallableData>;
    using pgExceptionRecord = Record<pgEmptyData>;
    using PathTracingSBT = ShaderBindingTable<pgRaygenRecord, pgMissRecord, pgHitgroupRecord, pgCallableRecord, pgExceptionRecord>;

    using ShapePtr = std::shared_ptr<Shape>;
    using MaterialPtr = std::shared_ptr<Material>;
    using TexturePtr = std::shaerd_ptr<Texture>;

    std::vector<ShapePtr> m_shapes;
    std::vector<std::variants<MaterialPtr, AreaEmitter>> m_surfaces;

    EnvironmentEmitter m_env;

    PathTracingSBT m_sbt;
};

} // ::prayground