#pragma once

#include <prayground/core/scene.h>

#include <tinyusdz/src/io-util.hh>
#include <tinyusdz/src/tinyusdz.hh>
#include <tinyusdz/src/tydra/render-data.hh>
#include <tinyusdz/src/tydra/scene-access.hh>
#include <tinyusdz/src/tydra/shader-network.hh>
#include <tinyusdz/src/usdShade.hh>
#include <tinyusdz/src/pprinter.hh>
#include <tinyusdz/src/prim-pprint.hh>
#include <tinyusdz/src/value-pprint.hh>
#include <tinyusdz/src/value-types.hh>

namespace prayground {

    namespace usd {

        bool MeshVisitor(const tinyusdz::Path& abs_path, const tinyusdz::Prim& prim, 
            const int32_t level, void* userdata, std::string* err);

        template <DerivedFromCamera _CamT, uint32_t _NRay>
        inline bool loadUSDToScene(const std::filesystem::path& filepath, Scene<_CamT, _NRay>& scene)
        {

        }

    } // namespace usd;

} // namespace prayground