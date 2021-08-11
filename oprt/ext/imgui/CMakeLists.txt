add_library( imgui STATIC  
    # Standards
    imconfig.h
    imgui_demo.cpp 
    imgui_draw.cpp 
    imgui_internal.h 
    imgui_tables.cpp 
    imgui_widgets.cpp 
    imgui.cpp 
    imgui.h 
    imstb_rectpack.h 
    imstb_textedit.h 
    imstb_truetype.h 

    # Backends 
    imgui_impl_opengl3.h 
    imgui_impl_opengl3.cpp 
    imgui_impl_glfw.h 
    imgui_impl_glfw.cpp
)

target_include_directories(imgui INTERFACE .. PRIVATE .)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui glfw glad ${OPENGL_gl_LIBRARY})