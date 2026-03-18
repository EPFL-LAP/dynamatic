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

add_llvm_executable(hls-fuzzer
        hls-fuzzer.cpp
        AbstractTarget.cpp
        AbstractWorker.cpp
        Options.cpp
        OptionsParser.cpp
        TargetRegistry.cpp
        targets/DynamaticTypeSystem.cpp
        targets/RandomCTarget.cpp
)
llvm_update_compile_flags(hls-fuzzer)
target_link_libraries(hls-fuzzer PRIVATE MLIRSupport DynamaticHLSFuzzer)
target_include_directories(hls-fuzzer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(hls-fuzzer DynamaticHLSFuzzerOptsTableGen)
