add_executable(empty-app 
    main.cpp 
    app.cpp 
    app.h
)

target_compile_definitions(
    empty-app
    PRIVATE
    OPRT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
    OPRT_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries( 
    empty-app oprt
)