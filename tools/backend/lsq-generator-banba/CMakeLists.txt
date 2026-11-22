set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(lsq-generator-banba
  lsq-generator-banba.cpp
)

llvm_update_compile_flags(lsq-generator-banba)
target_include_directories(lsq-generator-banba PRIVATE ${CIRCT_INCLUDE_DIRS})
target_link_libraries(lsq-generator-banba
  PRIVATE
  CIRCTHW

  MLIRIR
  MLIRSupport
  )
