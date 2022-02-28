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

using pgRGRecord = Record<pgRaygenData>;
using pgHGRecord = Record<pgHitgroupData>;
using pgMSRecord = Record<pgMissData>;
using pgCARecord = Record<pgCallableData>;
using pgEXRecord = Record<pgEmptyData>;
using pgSBT = ShaderBindingTable<pgRaygenRecord, pgMissRecord, pgHitgroupRecord, pgCallableRecord, pgExceptionRecord>;

class pgScene {
public:
    void createHitGroupSBT();

    void addShape(const std::shared_ptr<Shape>& shape);
private:
    using ShapePtr = std::shared_ptr<Shape>;
    using MaterialPtr = std::shared_ptr<Material>;
    using TexturePtr = std::shaerd_ptr<Texture>;

    std::vector<ShapePtr> m_shapes;
    std::vector<std::variants<MaterialPtr, AreaEmitter>> m_surfaces;

    EnvironmentEmitter m_env;
    pgPathTracingSBT m_sbt;
    pgPipeline m_pipeline;
};

} // ::prayground