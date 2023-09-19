add_subdirectory(generators)
add_llvm_executable(exp-export-vhdl export-vhdl.cpp)
llvm_update_compile_flags(exp-export-vhdl)
target_link_libraries(exp-export-vhdl PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  CIRCTHandshake
  CIRCTHW
  CIRCTESI
  )
  