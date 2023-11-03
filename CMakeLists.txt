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

include(FindPackageHandleStandardArgs)

#-------------------------------------------------------------------------------
# CIRCT/MLIR/LLVM configuration
#-------------------------------------------------------------------------------

find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
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
include_directories(${CIRCT_INCLUDE_DIRS})

option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

#-------------------------------------------------------------------------------
# Dynamatic configuration 
#-------------------------------------------------------------------------------

option(DYNAMATIC_BUILD_VISUALIZER "Disable the vivado synthesis tests." OFF)

# Define some variables for Dynamatic
set(DYNAMATIC_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(DYNAMATIC_INCLUDE_DIR ${DYNAMATIC_SOURCE_DIR}/include)
set(DYNAMATIC_BINARY_DIR ${CMAKE_BINARY_DIR})
set(DYNAMATIC_TOOLS_DIR ${DYNAMATIC_BINARY_DIR}/bin)
set(DYNAMATIC_EXPERIMENTAL_DIR ${DYNAMATIC_SOURCE_DIR}/experimental)
set(DYNAMATIC_BIN_INCLUDE_DIR ${DYNAMATIC_BINARY_DIR}/include)

# Add our own directory to the list of search paths for CMake modules 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Dynamatic include directories
include_directories(${DYNAMATIC_INCLUDE_DIR})
include_directories(${DYNAMATIC_BIN_INCLUDE_DIR})
include_directories(${DYNAMATIC_SOURCE_DIR}/tutorials/include)
include_directories(${DYNAMATIC_BINARY_DIR}/tutorials/include)
include_directories(${DYNAMATIC_EXPERIMENTAL_DIR}/include)
include_directories(${DYNAMATIC_BINARY_DIR}/experimental/include)

add_custom_target(dynamatic-headers)
set_target_properties(dynamatic-headers PROPERTIES FOLDER "Misc")

function(add_dynamatic_library name)
  add_mlir_library(${ARGV})
  add_dynamatic_library_install(${name})
endfunction()

function(add_dynamatic_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT DynamaticTargets)
  set_property(GLOBAL APPEND PROPERTY DYNAMATIC_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY DYNAMATIC_EXPORTS ${name})
endfunction()

function(add_dynamatic_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY DYNAMATIC_DIALECT_LIBS ${name})
  add_dynamatic_library(${ARGV} DEPENDS dynamatic-headers)
endfunction()

#-------------------------------------------------------------------------------
# Gurobi setup
#-------------------------------------------------------------------------------

find_package(GUROBI)
if (GUROBI_FOUND)
  message(STATUS "Found Gurobi")
  include_directories(${GUROBI_INCLUDE_DIRS})
else()
  message(WARNING "Gurobi wasn't found, the buffer placement infrastructure \
                   that requires solving an MILP will not be built with the \
                   project.")
  add_compile_definitions(DYNAMATIC_GUROBI_NOT_INSTALLED)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(${DYNAMATIC_INCLUDE_DIR}/dynamatic)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/experimental)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/lib)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/test)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/tools)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/tutorials)
