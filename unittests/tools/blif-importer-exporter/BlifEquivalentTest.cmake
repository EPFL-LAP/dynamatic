function(add_blif_equiv_test)
  set(options)
  set(oneValueArgs NAME INPUT BLIF2MLIR MLIR2BLIF MODE)
  set(multiValueArgs)
  cmake_parse_arguments(BET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT BET_NAME)
    message(FATAL_ERROR "add_blif_equiv_test requires NAME")
  endif()
  if(NOT BET_INPUT)
    message(FATAL_ERROR "add_blif_equiv_test requires INPUT")
  endif()
  if(NOT BET_BLIF2MLIR)
    message(FATAL_ERROR "add_blif_equiv_test requires BLIF2MLIR")
  endif()
  if(NOT BET_MLIR2BLIF)
    message(FATAL_ERROR "add_blif_equiv_test requires MLIR2BLIF")
  endif()
  if(NOT BET_MODE)
    set(BET_MODE combinational)
  endif()

  if(BET_MODE STREQUAL "combinational")
    set(ABC_COMMAND "cec")
  elseif(BET_MODE STREQUAL "sequential")
    set(ABC_COMMAND "dsec")
  else()
    message(FATAL_ERROR "MODE must be 'combinational' or 'sequential'")
  endif()

  # Resolve target to file path if it's a CMake target, otherwise use as-is
  if(TARGET ${BET_BLIF2MLIR})
    set(BLIF2MLIR_PATH "$<TARGET_FILE:${BET_BLIF2MLIR}>")
  else()
    set(BLIF2MLIR_PATH "${BET_BLIF2MLIR}")
  endif()

  if(TARGET ${BET_MLIR2BLIF})
    set(MLIR2BLIF_PATH "$<TARGET_FILE:${BET_MLIR2BLIF}>")
  else()
    set(MLIR2BLIF_PATH "${BET_MLIR2BLIF}")
  endif()

  set(TEST_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/${BET_NAME}_equiv.cmake)

  # Use file(GENERATE ...) instead of file(WRITE ...) to support generator expressions
  file(GENERATE OUTPUT ${TEST_SCRIPT}
    CONTENT
"set(INPUT_BLIF \"${BET_INPUT}\")
set(BLIF2MLIR \"${BLIF2MLIR_PATH}\")
set(MLIR2BLIF \"${MLIR2BLIF_PATH}\")
set(ABC_EXECUTABLE \"${ABC_EXECUTABLE}\")
set(ABC_COMMAND \"${ABC_COMMAND}\")
set(TMP_MLIR \"${CMAKE_CURRENT_BINARY_DIR}/${BET_NAME}.mlir\")
set(OUTPUT_BLIF \"${CMAKE_CURRENT_BINARY_DIR}/${BET_NAME}_out.blif\")

message(STATUS \"ABC_EXECUTABLE in test function: \${ABC_EXECUTABLE}\")
message(STATUS \"ABC_COMMAND: \${ABC_COMMAND}\")
message(STATUS \"BET_INPUT: \${BET_INPUT}\")
message(STATUS \"BLIF2MLIR_PATH: \${BLIF2MLIR_PATH}\")
message(STATUS \"MLIR2BLIF_PATH: \${MLIR2BLIF_PATH}\")

execute_process(
  COMMAND \${BLIF2MLIR} \${TMP_MLIR} \${INPUT_BLIF}
  RESULT_VARIABLE RES1
)
if(NOT RES1 EQUAL 0)
  message(FATAL_ERROR \"blif2mlir failed\")
endif()

execute_process(
  COMMAND \${MLIR2BLIF} \${TMP_MLIR} \${OUTPUT_BLIF}
  RESULT_VARIABLE RES2
)
if(NOT RES2 EQUAL 0)
  message(FATAL_ERROR \"mlir2blif failed\")
endif()

set(ABC_FULL_COMMAND \"\${ABC_COMMAND} \${INPUT_BLIF} \${OUTPUT_BLIF}\")
execute_process(
  COMMAND \${ABC_EXECUTABLE} -c \${ABC_FULL_COMMAND}
  RESULT_VARIABLE RES3
  OUTPUT_VARIABLE ABC_OUT
  ERROR_VARIABLE ABC_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)
message(STATUS \"ABC output:\n\${ABC_OUT}\")
message(STATUS \"ABC stderr:\n\${ABC_ERR}\")
if(NOT RES3 EQUAL 0)
  message(FATAL_ERROR \"ABC equivalence command failed\")
endif()
if(NOT ABC_OUT MATCHES \"Networks are equivalent\")
  message(FATAL_ERROR \"Circuits are NOT equivalent\")
endif()
message(STATUS \"Equivalence test PASSED\")
")

  add_test(
    NAME ${BET_NAME}
    COMMAND ${CMAKE_COMMAND} -P ${TEST_SCRIPT}
  )

  # Ensure the binary is built before the test runs
  if(TARGET ${BET_BLIF2MLIR})
    set_tests_properties(${BET_NAME} PROPERTIES DEPENDS ${BET_BLIF2MLIR})
  endif()

endfunction()
