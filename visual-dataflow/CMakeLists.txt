include_directories(${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/gen/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/gdextension)

add_subdirectory(godot-cpp)

# Set Godot lib name
if(CMAKE_BUILD_TYPE MATCHES Release)
  set(GODOT_LIB_NAME "VisualDataflow.linux.release.x86_64")
else()
  set(GODOT_LIB_NAME "VisualDataflow.linux.debug.x86_64")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/")

add_library(${GODOT_LIB_NAME}
  SHARED

  src/CSVParser.cpp
  src/DOTParser.cpp
  src/DOTReformat.cpp
  src/Graph.cpp
  src/GraphEdge.cpp
  src/GraphNode.cpp
  src/GraphParser.cpp
  src/RegisterTypes.cpp
  src/VisualDataflow.cpp
  src/MLIRMapper.cpp
)

target_link_libraries(${GODOT_LIB_NAME}
  PRIVATE
  DynamaticSupport
  DynamaticHandshake

  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect

  godot::cpp
)
