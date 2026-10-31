add_executable(
  property-expr-unit-tests
  PropertyExprTest.cpp
)
target_link_libraries(
  property-expr-unit-tests
  PRIVATE
  DynamaticExperimentalSupport
  GTest::gtest_main
)

enable_testing()

include(GoogleTest)
gtest_discover_tests(
  property-expr-unit-tests
)

add_custom_target(
  run-property-expr-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --timeout 1500 --output-on-failure
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Run the unit tests of the expressions of formal properties."
  VERBATIM
  USES_TERMINAL
  DEPENDS property-expr-unit-tests
)
add_to_unit_testing(run-property-expr-tests)
