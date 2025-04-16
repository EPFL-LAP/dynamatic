include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/2ae29b52fdff88c52fef655fa0d245fc514ca35b.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  integration
  TEST_SUITE.cpp
  memory/TEST_SUITE.cpp
  util.cpp
)
target_link_libraries(
  integration
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(integration)

add_definitions(-DDYNAMATIC_ROOT="${CMAKE_SOURCE_DIR}")
