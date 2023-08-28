add_llvm_executable(exp-constant constant.cpp)

llvm_update_compile_flags(exp-constant)
target_link_libraries(exp-constant PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  )
  