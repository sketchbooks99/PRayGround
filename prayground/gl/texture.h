#pragma once 

#include <glad/glad.h>
#include <prayground/core/util.h>
#include <prayground/math/vec.h>

namespace prayground {

    namespace gl {
        
        int32_t getNumChannelsOfTextureFormat(const int32_t format);

        class Texture {
        public:
            struct Settings
            {
                int32_t width;
                int32_t height;
                int32_t channels;

                int32_t dataType;
                int32_t internalFormat;
                int32_t format;

                int32_t magFilter;
                int32_t minFilter;

                int32_t wrapVertical;
                int32_t wrapHorizontal;

                Ver4f borderColor;
            };

            Texture();
            Texture(const Settings& settings, const uint8_t* data = nullptr);

            void loadData(const uint8_t* data, int32_t width, int32_t height);

            void draw() const;
            void draw(int32_t x, int32_t y) const;
            void draw(int32_t x, int32_t y, int32_t width, int32_t height) const;

        private:
            uint32_t m_texture;

            Settings m_settings;
        };

    } // namespace gl

} // namespace prayground