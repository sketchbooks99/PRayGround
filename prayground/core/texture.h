#pragma once 

#include <prayground/core/util.h>
#include <prayground/core/spectrum.h>

#ifndef __CUDACC__
    #include <map>
    #include <type_traits>
#endif

namespace prayground {

    enum class TextureType : uint32_t {
        None = 0,
        Constant = 1 << 0,
        Checkerboard = 1 << 1,
        Bitmap = 1 << 2,
        Custom = 1 << 3
    };

#ifndef __CUDACC__
    inline std::ostream& operator<<(std::ostream& out, TextureType type)
    {
        switch (type)
        {
        case TextureType::None:         return out << "TextureType::None";
        case TextureType::Constant:     return out << "TextureType::Constant";
        case TextureType::Checkerboard: return out << "TextureType::Checkerboard";
        case TextureType::Bitmap:       return out << "TextureType::Bitmap";
        case TextureType::Custom:       return out << "TextureType::Custom";
        }
    }
#endif

    class Texture {
    public:

        struct Data {
            void* data;

            /* SBT index of direct/continuation callable program for texture fetch */
            int32_t prg_id;
        };

#ifndef __CUDACC__
        Texture() = default;
        Texture(int prg_id) : m_prg_id(prg_id) {}

        virtual constexpr TextureType type() = 0;

        // Preparing texture data on the device.
        virtual void copyToDevice() = 0;
        virtual void free()
        {
            if (d_data) CUDA_CHECK(cudaFree(d_data));
            d_data = nullptr;
        }

        // Get data pointer on the device.
        void* devicePtr() const { return d_data; }

        Data getData() const { return { d_data, m_prg_id }; }

        void setProgramId(const int32_t prg_id) { m_prg_id = prg_id; }
        int32_t programId() const { return m_prg_id; }

    protected:
        void* d_data{ nullptr };
        int32_t m_prg_id;
#endif
    };

} // namespace prayground
