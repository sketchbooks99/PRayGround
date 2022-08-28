#pragma once

#include <prayground/core/util.h>

namespace prayground {
    
    namespace gl {
        
        // Draw rectangle which fills the whole window
        void drawRectangle();
        // Draw rectangle in a part of display
        void drawRectangle(int32_t x, int32_t y, int32_t w, int32_t h);
        
    } // namespace gl

} // namespace prayground
