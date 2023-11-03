add_subdirectory(include/experimental)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

if (DYNAMATIC_BUILD_VISUALIZER)
  add_subdirectory(visual-dataflow)
endif()
