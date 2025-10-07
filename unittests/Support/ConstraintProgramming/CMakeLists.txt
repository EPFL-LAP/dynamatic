if (GUROBI_FOUND)
add_executable(
  test-constraint-programming
  CPTest.cpp
)

target_link_libraries(
  test-constraint-programming
  PUBLIC
  GTest::gtest_main
  ${GUROBI_CXX_LIBRARY}
  ${GUROBI_LIBRARY}

  LLVMSupport
)

enable_testing()

include(GoogleTest)

gtest_discover_tests(
  test-constraint-programming
  XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/unittests/Support/ConstraintProgramming/results
)

# To run this unit test:
# ```
# ninja run-constraint-programming-test
# ```
add_custom_target(
  run-constraint-programming-test
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --timeout 1500 --output-on-failure
  DEPENDS test-constraint-programming
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/Support/ConstraintProgramming
  COMMENT "Run unit tests on constraint programming api."
  VERBATIM
  USES_TERMINAL
)



endif()
