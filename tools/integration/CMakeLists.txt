include(FetchContent)
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP true
  URL https://github.com/google/googletest/archive/2ae29b52fdff88c52fef655fa0d245fc514ca35b.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

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

add_custom_target(
  run-all-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --timeout 500 --output-on-failure
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run integration tests."
  VERBATIM
  USES_TERMINAL
)

add_custom_target(
  run-ci-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --timeout 500 --output-on-failure --exclude-regex _NoCI
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run integration tests that aren't marked with '_NoCI'."
  VERBATIM
  USES_TERMINAL
)
