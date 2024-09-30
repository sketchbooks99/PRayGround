set(APP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(APP_BINARY_DIR "${APP_DIR}/bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${APP_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${APP_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${APP_BINARY_DIR}")

message(STATUS "RUNTIME_OUTPUT_DIRECTORY: ${RUNTIME_OUTPUT_DIRECTORY}")

set(APP_PTX_DIR "${APP_BINARY_DIR}/ptx")
if (WIN32)
    string(REPLACE "/" "\\\\" APP_PTX_DIR ${APP_PTX_DIR})
endif()

set(CUDA_GENERATED_OUTPUT_DIR "${APP_BINARY_DIR}/ptx")

PRAYGROUND_add_executable(rtcamp10 target_name
    main.cpp 
    app.cpp 
    app.h
    kernels.cu
    params.h
)

target_include_directories( ${target_name} PRIVATE
    "${PRAYGROUND_DIR}"
    "${APP_BINARY_DIR}"
    "${OptiX_INCLUDE}"
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${APP_DIR}"
    APP_BINARY_DIR="${APP_BINARY_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})