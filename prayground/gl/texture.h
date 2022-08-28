#pragma once 

#include <prayground/gl/util.h>
#include <prayground/core/util.h>
#include <prayground/math/vec.h>

namespace prayground {

    namespace gl {
        
        int32_t getNumChannelsOfTextureFormat(const int32_t format);

        class Texture {
        public:
            struct Settings
            {
                /* Texture data dimensions */
                int32_t width           = 256;
                int32_t height          = 256;
                int32_t channels        = 3;

                /* The data type of texture: GL_UNSIGNED_BYTE, GL_FLOAT, ...*/
                int32_t dataType        = GL_UNSIGNED_BYTE;
                /* Internal format of texture: GL_RGBA32F, GL_R8, ... */
                int32_t internalFormat  = GL_RGB8;
                /* Texture format: GL_RED, GL_RGBA, ... */
                int32_t format          = GL_RGB;

                /* Texture filtering: GL_NEAREST, GL_LINEAR */
                int32_t minFilter       = GL_LINEAR;
                int32_t magFilter       = GL_LINEAR;

                /* Texture wrapping: GL_REPEAT, GL_CLAMP_TO_EDGE, ... */
                int32_t wrapVertical    = GL_CLAMP_TO_EDGE;
                int32_t wrapHorizontal  = GL_CLAMP_TO_EDGE;

                /* Border color of texture */
                Vec4f borderColor       = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
            };

            Texture() = default;
            Texture(const Settings& settings, const uint8_t* data = nullptr);

            void allocate(int32_t width, int32_t height, int32_t internal_format);
            void allocate(const Settings& settings);
            
            void bind();
            void unbind();
            
            /**
             * @brief Load pixel data to texture
             * 
             * @param data : Pixel data
             * @param width : Width of input image
             * @param height : Height of input image
             * @param internal_format : TInternal format of texture: GL_RGBA32F, GL_R8, ...
             * @param data_type : The data type of texture: GL_UNSIGNED_BYTE, GL_FLOAT, ...
             */
            void loadData(const uint8_t* data, int32_t width, int32_t height, int32_t internal_format, int32_t data_type);

            void draw() const;
            void draw(int32_t x, int32_t y) const;
            void draw(int32_t x, int32_t y, int32_t width, int32_t height) const;

        private:
            uint32_t m_texture;

            Settings m_settings;
        };

    } // namespace gl

} // namespace prayground