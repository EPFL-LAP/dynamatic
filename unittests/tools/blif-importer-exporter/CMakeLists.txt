# Testing using the BLIF files contained in test-cases which should contain a variety of combinational and sequential circuits

get_filename_component(DYNAMATIC_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
set(AIG_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test-cases)
set(BIN_DIR ${DYNAMATIC_BASE_DIR}/bin)
set(ABC_EXECUTABLE ${CMAKE_BINARY_DIR}/_deps/abc-build/abc)

if(DYNAMATIC_ENABLE_ABC)
  include(BlifEquivalentTest.cmake)
  enable_testing()

  file(GLOB_RECURSE BLIF_TEST_FILES
    "${AIG_DATA_DIR}/*.blif"
  )

  foreach(BLIF_FILE ${BLIF_TEST_FILES})
    # Skip yosys files
    if(BLIF_FILE MATCHES "_yosys\\.blif$")
      continue()
    endif()
    # Create a unique test name from the file path
    file(RELATIVE_PATH REL_PATH
      "${AIG_DATA_DIR}"
      ${BLIF_FILE}
    )
    string(REPLACE "/" "_" TEST_NAME ${REL_PATH})
    string(REPLACE ".blif" "" TEST_NAME ${TEST_NAME})

    # Check if the blif contains any .latch lines to determine if it's combinational or sequential
    file(READ ${BLIF_FILE} BLIF_CONTENT)
    if(BLIF_CONTENT MATCHES "\\.latch[ \t]")
        set(BET_MODE sequential)
    else()
        set(BET_MODE combinational)
    endif()

    add_blif_equiv_test(
      NAME ${TEST_NAME}
      INPUT ${BLIF_FILE}
      BLIF2MLIR ${BIN_DIR}/import-blif
      MLIR2BLIF ${BIN_DIR}/export-blif
      MODE ${BET_MODE}
    )
  endforeach()

  add_custom_target(
    run-blif-equiv-tests
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --timeout 1500 --output-on-failure
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Run BLIF equivalence tests."
    VERBATIM
    USES_TERMINAL
  )
  add_to_unit_testing(check-dynamatic run-blif-equiv-tests)

endif()
