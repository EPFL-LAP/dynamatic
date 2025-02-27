configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(DYNAMATIC_TEST_DEPENDS
  FileCheck count not
  split-file
  dynamatic-opt
  )

add_lit_testsuite(check-dynamatic-xls "Running the Dynamatic XLS integration regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNAMATIC_TEST_DEPENDS}
  )
set_target_properties(check-dynamatic-xls PROPERTIES FOLDER "Tests")

add_lit_testsuites(DynamaticXls ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNAMATIC_TEST_DEPENDS}
  )
