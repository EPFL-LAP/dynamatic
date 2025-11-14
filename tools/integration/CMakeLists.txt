add_executable(
  integration
  TEST_SUITE.cpp
  util.cpp
)
target_link_libraries(
  integration
  GTest::gtest_main
  nlohmann_json::nlohmann_json
)

enable_testing()

include(GoogleTest)
gtest_discover_tests(
  integration
  XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/integration/results
)

add_definitions(-DDYNAMATIC_ROOT="${CMAKE_SOURCE_DIR}")

# Default: Execute 1 integration tests in parallel
# Can be overridden with -DPARALLEL_FACTOR=<n> at configure time
# e.g. by running 'cmake -DPARALLEL_FACTOR=<n> -B build'
# before 'ninja -C build <target>'
set(PARALLEL_FACTOR 1 CACHE STRING "Parallelism for integration test targets")

add_custom_target(
  run-all-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel ${PARALLEL_FACTOR} --timeout 1500 --output-on-failure
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run integration tests."
  VERBATIM
  USES_TERMINAL
)

add_custom_target(
  run-ci-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel ${PARALLEL_FACTOR} --timeout 1500 --output-on-failure --exclude-regex _NoCI
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run integration tests that aren't marked with '_NoCI'."
  VERBATIM
  USES_TERMINAL
)

add_custom_target(
  run-spec-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel ${PARALLEL_FACTOR} --timeout 1500 --output-on-failure --tests-regex spec
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run speculation integration tests."
  VERBATIM
  USES_TERMINAL
)
