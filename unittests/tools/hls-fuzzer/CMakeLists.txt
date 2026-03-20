add_executable(
  hls-fuzzer-unit-tests
  TEST_SUITE.cpp
)
target_link_libraries(
  hls-fuzzer-unit-tests
  PRIVATE
  DynamaticHLSFuzzer
  GTest::gtest_main
)

enable_testing()

include(GoogleTest)
gtest_discover_tests(
  hls-fuzzer-unit-tests
)

add_custom_target(
  run-hls-fuzzer-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --timeout 1500 --output-on-failure
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Run HLS Fuzzer tests."
  VERBATIM
  USES_TERMINAL
  DEPENDS hls-fuzzer-unit-tests
)
add_to_unit_testing(run-hls-fuzzer-tests)
