#include "usd_loader.h"

namespace prayground {

    namespace usd {

        using XformMap = std::map<std::string, const tinyusdz::Xform*>;
        using MeshMap = std::map<std::string, const tinyusdz::GeomMesh*>;
        using MaterialMap = std::map<std::string, const tinyusdz::Material*>;
        using PreviewSurfaceMap = std::map<std::string, std::pair<const tinyusdz::Shader*, const tinyusdz::UsdPreviewSurface*>>;
        using UVTextureMap = std::map<std::string, std::pair<const tinyusdz::Shader*, const tinyusdz::UsdUVTexture*>>;
        using PrimvarReader_float2Map = 
            std::map<std::string, std::pair<const tinyusdz::Shader*, const tinyusdz::UsdPrimvarReader_float2*>>;

        // @todo This scene is different from prayground::Scene, and will be used as input arg 'userdata' of MeshVisitor function.
        template <typename T>
        struct Keyframe {
            float time;
            T value;
        };

        template <typename T>
        struct Animation {
            std::string name;
            std::vector<Keyframe<T>> keyframes;
        };

        struct Scene {
            struct Object {
                std::shared_ptr<Shape> shape;
                std::vector<std::shared_ptr<Material>> materials;
                std::vector<Animation<Matrix4f>> animations;
            };

            struct Light {
                std::shared_ptr<Shape> shape;
                std::shared_ptr<AreaEmitter> emitter;
                std::vector<Animation<Matrix4f>> animations;
            };

            std::vector<Object> objects;
            std::vector<Light> lights;
        };

        bool MeshVisitor(const tinyusdz::Path& abs_path, const tinyusdz::Prim& prim, 
            const int32_t level, void* userdata, std::string* err)
        {
            if (!userdata)
                return false;

            if (level > 1024 * 1024)
            {
                if (err)
                    *err = "Too deep hierarchy.\n";
                return false;
            }

            if (const tinyusdz::GeomMesh* mesh = prim.as<tinyusdz::GeomMesh>())
            {
                const std::string mesh_path_str = abs_path.full_path_name();

                usd::Scene *scene = reinterpret_cast<usd::Scene*>(userdata);
            }

            return true;
        }

    } // namespace usd

} // namespace prayground