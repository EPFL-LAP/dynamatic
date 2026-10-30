configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(DYNAMATIC_TEST_DEPENDS
  FileCheck count not
  split-file
  dynamatic-opt
  export-rtl
  hls-fuzzer
  hls-fuzzer-check-bitwidth
  translate-llvm-to-std
  source-rewriter
  DynPragmasPlugin
  clang
  )

add_lit_testsuite(check-dynamatic "Running the Dynamatic regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNAMATIC_TEST_DEPENDS}
  )
set_target_properties(check-dynamatic PROPERTIES FOLDER "Tests")

add_lit_testsuites(Dynamatic ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNAMATIC_TEST_DEPENDS}
  )

include_directories(include)
include_directories(${DYNAMATIC_BINARY_DIR}/test/include)

add_subdirectory(include/test/Transforms)
add_subdirectory(lib)
