include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/2ae29b52fdff88c52fef655fa0d245fc514ca35b.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  integration
  TEST_SUITE.cpp
  util.cpp
)
target_link_libraries(
  integration
  GTest::gtest_main
)

enable_testing()

include(GoogleTest)
gtest_discover_tests(integration)

add_definitions(-DDYNAMATIC_ROOT="${CMAKE_SOURCE_DIR}")

add_custom_target(run-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --output-on-failure
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run integration tests."
  VERBATIM
)
