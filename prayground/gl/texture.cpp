#include "texture.h"

namespace prayground {

    namespace gl {

        struct GLFormatToChannels 
        {
            int32_t operator[](const int32_t format)
            {
                switch(format)
                {
                case GL_RED:
                case GL_R8:
                case GL_R32F:
                    return 1; 
                case GL_RG:
                case GL_RG8:
                case GL_RG32F:
                    return 2;
                case GL_RGB:
                case GL_RGB8:
                case GL_RGB32F:
                    return 3;
                case GL_RGBA:
                case GL_RGBA8:
                case GL_RGBA32F:
                    return 4;
                }
                return 0;
            }
        };

        int32_t getNumChannelsOfTextureFormat(const int32_t format)
        {

        }

        Texture::Texture()
        {

        }

        Texture::Texture(const Texture::Settings& settings, const uint8_t* data)
        {

        }

        void Texture::loadData(const uint8_t* data, int32_t width, int32_t height, int32_t internal_format, int32_t data_type)
        {
            glBindTexture(GL_TEXTURE_2D, m_texture);

            GLuint format = GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, data_type, data);
        }

        void Texture::draw() const 
        {

        }

        void Texture::draw(int32_t x, int32_t y) const
        {

        }

        void Texture::draw(int32_t x, int32_t y, int32_t width, int32_t height) const
        {

        }

    } // namespace gl

} // namespace prayground