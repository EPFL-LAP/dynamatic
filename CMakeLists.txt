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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -frtti")
endif ()

include(FindPackageHandleStandardArgs)
find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)
find_package(Boost REQUIRED COMPONENTS)

# Build Dynamatic with ccache if the package is present
set(DYNAMATIC_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(DYNAMATIC_CCACHE_BUILD)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    else()
        message(WARNING "ccache not found, but DYNAMATIC_CCACHE_BUILD is ON")
    endif()
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM configuration
#-------------------------------------------------------------------------------

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${CLANG_DIR}")
message(STATUS "Using PollyConfig.cmake in: ${Polly_DIR}")

find_package(MLIR REQUIRED CONFIG)
set(Clang_DIR ${CLANG_DIR})
find_package(Clang REQUIRED CONFIG)
find_package(Polly REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)
include(HandleLLVMOptions)
include(FetchContent)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

link_directories(${Polly_LIBRARIES})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${Polly_INCLUDE_DIRS})
message(STATUS "Including from ${Polly_INCLUDE_DIRS}")

option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

#-------------------------------------------------------------------------------
# Dynamatic configuration 
#-------------------------------------------------------------------------------

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
include_directories(DYNAMATIC_BINARY_DIR ${CMAKE_BINARY_DIR})
include_directories(${DYNAMATIC_BIN_INCLUDE_DIR})
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

function(add_dynamatic_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(dynamatic-headers MLIR${dialect}IncGen)
endfunction()

function(add_dynamatic_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(dynamatic-headers MLIR${interface}IncGen)
endfunction()

#-------------------------------------------------------------------------------
# Gurobi setup
#-------------------------------------------------------------------------------

find_package(GUROBI)
if (GUROBI_FOUND)
  include_directories(${GUROBI_INCLUDE_DIRS})
else()
  message(WARNING "Gurobi wasn't found, the buffer placement algorithms \
                   that requires solving MILPs will not be built with the \
                   project.")
  add_compile_definitions(DYNAMATIC_GUROBI_NOT_INSTALLED)
endif()

#-------------------------------------------------------------------------------
# CBC solver setup
#-------------------------------------------------------------------------------

# Find the CBC, Osi, Clp libraries
find_package(PkgConfig REQUIRED)

# Use pkg-config to get include dirs and libs
# This command sets 
# - *_INCLUDE_DIRS
# - *_LIBRARY_DIRS
# But these variable will not be automatically discovered by the target.
pkg_check_modules(CBC REQUIRED cbc)
pkg_check_modules(CLPCORE REQUIRED clp)
pkg_check_modules(OSI REQUIRED osi)

# This command makes the include directory of CBC discoverable
include_directories(dynamatic-opt PRIVATE
  ${CBC_INCLUDE_DIRS}
  ${CLPCORE_INCLUDE_DIRS}
  ${OSI_INCLUDE_DIRS}
)

# This command makes the link directory of CBC discoverable
# Without this, -lCbc would not work on some distros
link_directories(dynamatic-opt PRIVATE
  ${CBC_LIBRARY_DIRS}
  ${CLPCORE_LIBRARY_DIRS}
  ${OSI_LIBRARY_DIRS}
)

#-------------------------------------------------------------------------------
# XLS Integration (experimental)
#-------------------------------------------------------------------------------

set(DYNAMATIC_XLS_DIR ${DYNAMATIC_SOURCE_DIR}/xls)

if(DYNAMATIC_ENABLE_XLS)
  # Add 'DYNAMATIC_ENABLE_XLS` macro to prepoc + tablegen for conditioanl
  # inclusion of XLS features.
  add_compile_definitions(DYNAMATIC_ENABLE_XLS)
  set(LLVM_TABLEGEN_FLAGS ${LLVM_TABLEGEN_FLAGS} -DDYNAMATIC_ENABLE_XLS)

  include_directories(${DYNAMATIC_EXPERIMENTAL_DIR}/xls/include)
  include_directories(${DYNAMATIC_BINARY_DIR}/experimental/xls/include)
  include_directories(${DYNAMATIC_XLS_DIR})
endif()


#-------------------------------------------------------------------------------
# External binaries
#-------------------------------------------------------------------------------

if(DYNAMATIC_ENABLE_LEQ_BINARIES)
  # Download modified NuSMV binary,
  # which support to print larger sets of reachable states
  set(EXT_BIN_DIR "${DYNAMATIC_SOURCE_DIR}/ext")
  file(DOWNLOAD
      https://github.com/ETHZ-DYNAMO/NuSMV-2.7.0/releases/download/v1.0.1/NuSMV
      ${EXT_BIN_DIR}/NuSMV
  )
  file(CHMOD ${EXT_BIN_DIR}/NuSMV PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

  # Download dot2smv Python scripts
  FetchContent_Declare(
    dot2smv
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR     ${EXT_BIN_DIR}/dot2smv
    URL https://github.com/Jiahui17/dot2smv/archive/refs/heads/elastic-miter.tar.gz
  )

  FetchContent_MakeAvailable(dot2smv)
endif()

# We use GoogleTest for regression tests
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP true
  URL https://github.com/google/googletest/archive/2ae29b52fdff88c52fef655fa0d245fc514ca35b.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# TODO: Use the JSON package from Boost instead.
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(${DYNAMATIC_INCLUDE_DIR}/dynamatic)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/experimental)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/lib)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/test)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/tools)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/tutorials)
add_subdirectory(${DYNAMATIC_SOURCE_DIR}/unittests)
