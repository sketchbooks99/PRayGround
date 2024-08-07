find_package( OpenGL REQUIRED ) 

if (BUILD_SHARED_LIBS AND UNIX)
    set(GLFW_LIB_NAME glfw)
else()
    set(GLFW_LIB_NAME glfw3)
endif()

# Filter out warnings that cause problems with GLFW.
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  string( REPLACE "-Wdeclaration-after-statement" "" filtered_c_flags ${CMAKE_C_FLAGS} )
  string( REPLACE "-Wsign-compare" "" filtered_c_flags ${filtered_c_flags} )
  push_variable( CMAKE_C_FLAGS "${filtered_c_flags} -Wno-format-truncation -Wno-deprecated" )
endif()

add_subdirectory(glfw)
set_property(TARGET glfw PROPERTY C_STANDARD 99)
if (WARNING_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
    pop_variable(CMAKE_C_FLAGS)
endif()

add_subdirectory(imgui)

add_library(glad SHARED 
    KHR/khrplatform.h 
    glad/glad.c 
    glad/glad.h
)
target_compile_definitions( glad 
    PRIVATE GLAD_GLAPI_EXPORT_BUILD
    PUBLIC GLAD_GLAPI_EXPORT
)
target_include_directories(glad PUBLIC .)
target_link_libraries(glad PUBLIC ${OPENGL_LIBRARIES})

add_subdirectory(nanovdb)

set_property( TARGET glad PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET glfw PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET imgui PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )