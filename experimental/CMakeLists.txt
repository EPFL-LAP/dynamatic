include_directories(include)
include_directories(${DYNAMATIC_BINARY_DIR}/experimental/include)

add_subdirectory(include/experimental)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
