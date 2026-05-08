add_executable(
  blif-file-manager-unit-tests
  BLIFFileManagerTest.cpp
)

target_link_libraries(
  blif-file-manager-unit-tests
  PRIVATE
  DynamaticSupport
  DynamaticAnalysis
  DynamaticSynth
  MLIRControlFlowDialect
  MLIRMathDialect
  MLIRSCFDialect
  GTest::gtest_main
)

target_compile_definitions(
  blif-file-manager-unit-tests
  PRIVATE
  BLIF_TEST_DIR="${CMAKE_SOURCE_DIR}/aig_folder/"
  DYNAMATIC_ROOT="${CMAKE_SOURCE_DIR}/"
  RTL_JSON_FILE="${CMAKE_SOURCE_DIR}/data/rtl-config-verilog-beta.json"
)

enable_testing()

include(GoogleTest)

gtest_discover_tests(
  blif-file-manager-unit-tests
  XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/unittests/Support/BLIFFileManager/results
)

add_custom_target(
  run-blif-file-manager-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --timeout 1500 --output-on-failure
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Run BLIFFileManager unit tests."
  VERBATIM
  USES_TERMINAL
  DEPENDS blif-file-manager-unit-tests
)

add_to_unit_testing(check-dynamatic run-blif-file-manager-tests)
