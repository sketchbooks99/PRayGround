#pragma once 

#include <glad/glad.h>

namespace prayground {

    namespace gl {

        /** Frame buffer object */
        class Fbo {
        public:
            struct Settings 
            {
                
            };

            Fbo();

            void allocate(const Settings& settings);
            void allocate(int32_t width, int32_t height);

            void begin();
            void end();

        private:
            uint32_t mColorTexture;
            uint32_t mDepthTexture;
            uint32_t mStencilTexture;
        };

    } // namespace prayground

} // namespace prayground