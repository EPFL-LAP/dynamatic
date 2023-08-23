add_llvm_executable(exp-cmpi cmpi.cpp)

llvm_update_compile_flags(exp-cmpi)
target_link_libraries(exp-cmpi PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect

  CIRCTHandshake
  CIRCTHW
  CIRCTESI
  )