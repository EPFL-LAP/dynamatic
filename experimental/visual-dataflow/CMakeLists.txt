include_directories(${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/gen/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/gdextension)

add_subdirectory(godot-cpp)

set(GODOT_LIB_NAME "visual_dataflow.linux.template_debug.x86_64")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/visual-dataflow/bin/")

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
 
  AffineToScf
  DynamaticLowerScfToCf
  DynamaticBufferPlacement
  DynamaticStandardToHandshakeFPGA18
  DynamaticHandshakeToNetlist
  DynamaticTransforms
  DynamaticTutorialsCreatingPasses
  DynamaticExperimentalTestTransforms
  DynamaticExperimentalSupport

  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTStandardToHandshake

  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect

  godot::cpp
)
