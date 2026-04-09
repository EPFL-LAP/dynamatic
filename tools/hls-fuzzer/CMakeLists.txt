include(TableGen)

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(DynamaticHLSFuzzerOptsTableGen)

set(LLVM_LINK_COMPONENTS
        Support Option
)

add_llvm_library(DynamaticHLSFuzzer
        AST.cpp
        BasicCGenerator.cpp
        Randomly.cpp
        TypeSystem.cpp
        PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(DynamaticHLSFuzzer)
target_link_libraries(DynamaticHLSFuzzer PUBLIC MLIRSupport)
target_include_directories(DynamaticHLSFuzzer INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

add_llvm_executable(hls-fuzzer
        hls-fuzzer.cpp
        AbstractTarget.cpp
        AbstractWorker.cpp
        Options.cpp
        OptionsParser.cpp
        TargetRegistry.cpp
        targets/BitwidthOptimizationsTarget.cpp
        targets/BitwidthTypeSystem.cpp
        targets/DynamaticTypeSystem.cpp
        targets/LSQNoDepTypeSystem.cpp
        targets/LSQOptimizationsTarget.cpp
        targets/RandomCTarget.cpp
        targets/TargetUtils.cpp
)
llvm_update_compile_flags(hls-fuzzer)
target_link_libraries(hls-fuzzer PRIVATE MLIRSupport DynamaticHLSFuzzer)
target_include_directories(hls-fuzzer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
# Also keep dynamatic up-to-date such that we are always testing the up-to-date verison.
add_dependencies(hls-fuzzer DynamaticHLSFuzzerOptsTableGen dynamatic)

add_subdirectory(oracles)
