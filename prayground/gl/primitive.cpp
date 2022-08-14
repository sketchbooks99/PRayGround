#include "primitive.h"
#include <prayground/gl/util.h>
#include <prayground/app/app_runner.h>

namespace prayground {

    namespace gl {

        struct BufferObjects {
            void generate()
            {
                glGenVertexArrays(1, &vao);
                glGenBuffers(1, &vbo);
                glGenBuffers(1, &ebo);
            }

            /* Vertex buffer object */
            GLuint vbo;
            
            /* Vertex array object */
            GLuint vao;

            /* Element buffer object */
            GLuint ebo;
        };

        class Rectangle {
        public:
            void init()
            {
                m_buffers.generate();

                glBindBuffer(GL_ARRAY_BUFFER, m_buffers.vbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 20, nullptr, GL_DYNAMIC_DRAW);

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffers.ebo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(GLuint) * 6, nullptr, GL_DYNAMIC_DRAW);

                m_vertices = {
                    // position             texcoord (vertically flipped)
                    -1.0f, -1.0f, 0.0f,     0.0f, 1.0f,
                    -1.0f, 1.0f, 0.0f,      0.0f, 0.0f,
                    1.0f, -1.0f, 0.0f,      1.0f, 1.0f,
                    1.0f, 1.0f, 0.0f,       1.0f, 0.0f
                };

                m_indices = { 
                    0, 1, 2,  
                    2, 1, 3 
                };

                is_initialized = true;
            }
            
            void draw(float x0, float x1, float y0, float y1)
            {
                m_vertices[0] = x0;  m_vertices[1] = y0;
                m_vertices[5] = x0;  m_vertices[6] = y1;
                m_vertices[10] = x1; m_vertices[11] = y0;
                m_vertices[15] = x1; m_vertices[16] = y1;

                glBindVertexArray(m_buffers.vao);

                // Set vertices data to vertex buffer object
                glBindBuffer(GL_ARRAY_BUFFER, m_buffers.vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * m_vertices.size(), m_vertices.data());

                // Set indices data to element buffer object
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffers.ebo);
                glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(GLuint) * m_indices.size(), m_indices.data());

                // Enable 'position' attribute
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
                glEnableVertexAttribArray(0);

                // Enable 'texcoord' attribute
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
                glEnableVertexAttribArray(1);

                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            }

            const bool& IsInitialized() const
            {
                return is_initialized;
            }

        private:
            std::array<GLfloat, 20> m_vertices;
            std::array<GLuint, 6> m_indices;

            BufferObjects m_buffers;

            bool is_initialized = false;
        };

        static Rectangle g_rectangle;

        void drawRectangle()
        {
            GLfloat x0 = -1.0f, x1 = 1.0f, y0 = -1.0f, y1 = 1.0f;
            g_rectangle.draw(x0, x1, y0, y1);
        }

        void drawRectangle(int32_t x, int32_t y, int32_t w, int32_t h)
        {
            int32_t window_width = pgGetWidth(), window_height = pgGetHeight();
            GLfloat x0 = -1.0f + ((static_cast<float>(x) / window_width) * 2.0f);
            GLfloat x1 = -1.0f + ((static_cast<float>(x + w) / window_width) * 2.0f);
            GLfloat y0 = 1.0f - ((static_cast<float>(y + h) / window_height) * 2.0f);
            GLfloat y1 = 1.0f - ((static_cast<float>(y) / window_height) * 2.0f);

            g_rectangle.draw(x0, x1, y0, y1);
        }
    }

} // namespace prayground