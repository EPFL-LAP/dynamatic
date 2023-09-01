add_llvm_executable(exp-buffer buffer.cpp)

llvm_update_compile_flags(exp-buffer)
target_link_libraries(exp-buffer PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  )
  