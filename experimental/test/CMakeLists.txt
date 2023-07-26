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

add_lit_testsuite(check-dynamatic-experimental "Running the Dynamatic regression tests for experimental work"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNAMATIC_TEST_DEPENDS}
  )
set_target_properties(check-dynamatic-experimental PROPERTIES FOLDER "Tests")

add_lit_testsuites(DynamaticExperimental ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNAMATIC_TEST_DEPENDS}
  )

add_subdirectory(lib)
