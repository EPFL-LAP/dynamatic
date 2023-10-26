add_llvm_executable(exp-cmpf cmpf.cpp)

llvm_update_compile_flags(exp-cmpf)
target_link_libraries(exp-cmpf PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  )
