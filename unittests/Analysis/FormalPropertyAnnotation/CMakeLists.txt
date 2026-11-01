add_executable(
  formal-property-annotation-unit-tests
  EliminationTest.cpp
)
# The helpers of the pass are private headers next to its sources
target_include_directories(
  formal-property-annotation-unit-tests
  PRIVATE
  ${DYNAMATIC_EXPERIMENTAL_DIR}/lib/Analysis/FormalPropertyAnnotation
)
target_link_libraries(
  formal-property-annotation-unit-tests
  PRIVATE
  DynamaticFormalPropertyAnnotation
  GTest::gtest_main
)

enable_testing()

include(GoogleTest)
gtest_discover_tests(
  formal-property-annotation-unit-tests
)

add_custom_target(
  run-formal-property-annotation-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --timeout 1500 --output-on-failure
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Run the unit tests of the helpers of the formal property annotation."
  VERBATIM
  USES_TERMINAL
  DEPENDS formal-property-annotation-unit-tests
)
add_to_unit_testing(run-formal-property-annotation-tests)
