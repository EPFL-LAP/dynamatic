#-------------------------------------------------------------------------------
# CMake/C++ configuration
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)
project(Dynamatic)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GR-")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
endif ()

#-------------------------------------------------------------------------------
# MLIR/LLVM configuration
#-------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

# Dynamatic variables
set(DYNAMATIC_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)
set(DYNAMATIC_INCLUDE_DIR ${DYNAMATIC_SOURCE_DIR}/include)
set(DYNAMATIC_BINARY_DIR ${CMAKE_SOURCE_DIR}/../build)
set(DYNAMATIC_TOOLS_DIR ${DYNAMATIC_BINARY_DIR}/bin)
set(DYNAMATIC_EXPERIMENTAL_DIR ${DYNAMATIC_SOURCE_DIR}/experimental)
set(DYNAMATIC_BIN_INCLUDE_DIR ${DYNAMATIC_BINARY_DIR}/include)

# Dynamatic include directories
include_directories(${DYNAMATIC_INCLUDE_DIR})
include_directories(${DYNAMATIC_BIN_INCLUDE_DIR})
include_directories(${DYNAMATIC_SOURCE_DIR}/tutorials/include)
include_directories(${DYNAMATIC_BINARY_DIR}/tutorials/include)
include_directories(${DYNAMATIC_EXPERIMENTAL_DIR}/include)
include_directories(${DYNAMATIC_BINARY_DIR}/experimental/include)

# visual-dataflow variables
set(VISUAL_DATAFLOW_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)
set(GODOT_CPP_DIR ${CMAKE_SOURCE_DIR}/godot-cpp)

# visual-dataflow include directories
include_directories(${GODOT_CPP_DIR}/include)
include_directories(${GODOT_CPP_DIR}/gen/include)
include_directories(${GODOT_CPP_DIR}/gdextension)

# Build Godot C++ bindings
add_subdirectory(godot-cpp)

# Set Godot lib name and output directory
set(GODOT_LIB_NAME "VisualDataflow.linux.x86_64")

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
  ${DYNAMATIC_BINARY_DIR}/lib/libDynamaticSupport.a
  ${DYNAMATIC_BINARY_DIR}/lib/libDynamaticHandshake.a

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
