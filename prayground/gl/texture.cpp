#include "texture.h"
#include <prayground/app/app_runner.h>
#include <prayground/gl/primitive.h>

namespace prayground {

    namespace gl {

        int32_t getNumChannelsOfTextureFormat(const int32_t format)
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

        // --------------------------------------------------------------------
        Texture::Texture(const Texture::Settings& settings, const uint8_t* data)
        {
            allocate(settings);
            if (data != nullptr)
                loadData(data, settings.width, settings.height, settings.internalFormat, settings.dataType);
        }

        void Texture::allocate(int32_t width, int32_t height, int32_t internal_format)
        {
            Settings settings;
            settings.width = width;
            settings.height = height;
            settings.internalFormat = internal_format;
            // Determine data type and format based on the internal format.
            switch (internal_format)
            {
            case GL_R32F:
                settings.dataType = GL_FLOAT;
            case GL_RED:
            case GL_R8:
                settings.format = GL_RED;
                break;

            case GL_RG32F:
                settings.dataType = GL_FLOAT;
            case GL_RG:
            case GL_RG8:
                settings.format = GL_RG;
                break;

            case GL_RGB32F:
                settings.dataType = GL_FLOAT;
            case GL_RGB:
            case GL_RGB8:
                settings.format = GL_RGB;
                break;

            case GL_RGBA32F:
                settings.dataType = GL_FLOAT;
            case GL_RGBA:
            case GL_RGBA8:
                settings.format = GL_RGB;
                break;
            }

            allocate(settings);
        }

        void Texture::allocate(const Texture::Settings& settings)
        {
            m_texture = 0;
            glGenTextures(1, &m_texture);
            glBindTexture(GL_TEXTURE_2D, m_texture);

            // Set texture a filtering method
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)settings.minFilter);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)settings.magFilter);
            // Set texture a wrapping method
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)settings.wrapHorizontal);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)settings.wrapVertical);

            // Set border color 
            float border_color[4];
            settings.borderColor.toArray(border_color);
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);

            // Unbind texture
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        void Texture::bind()
        {
            glBindTexture(GL_TEXTURE_2D, m_texture);
        }

        void Texture::unbind()
        {
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // --------------------------------------------------------------------
        /// @todo : Validation to check if internal_format and data_type are correct 
        void Texture::loadData(const uint8_t* data, int32_t width, int32_t height, int32_t internal_format, int32_t data_type)
        {
            // Automatically determine the texture format
            GLuint format;
            switch(internal_format)
            {
                case GL_R32F:
                    // Make sure 'data_type' be GL_FLOAT 
                    data_type = GL_FLOAT;
                case GL_RED:
                case GL_R8:
                    format = GL_RED;
                    break;
                case GL_RG32F:
                    data_type = GL_FLOAT;
                case GL_RG:
                case GL_RG8:
                    format = GL_RG;
                    break;
                case GL_RGB32F:
                    data_type = GL_FLOAT;
                case GL_RGB:
                case GL_RGB8:
                    format = GL_RGB;
                    break;
                case GL_RGBA32F:
                    data_type = GL_FLOAT;
                case GL_RGBA:
                case GL_RGBA8:
                    format = GL_RGBA;
                    break;
            }

            // Update settings
            m_settings.width = width;
            m_settings.height = height;
            m_settings.internalFormat = internal_format;
            m_settings.format = format;
            m_settings.dataType = data_type;

            glBindTexture(GL_TEXTURE_2D, m_texture);
            // Set pixel data to texture
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, data_type, data);
            // Unbind texture
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // --------------------------------------------------------------------
        void Texture::draw() const 
        {
            draw(0, 0, m_settings.width, m_settings.height);
        }

        void Texture::draw(int32_t x, int32_t y) const
        {
            draw(x, y, m_settings.width, m_settings.height);
        }

        void Texture::draw(int32_t x, int32_t y, int32_t width, int32_t height) const
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_texture);

            drawRectangle(x, y, width, height);

            glBindTexture(GL_TEXTURE_2D, 0);
        }

    } // namespace gl

} // namespace prayground