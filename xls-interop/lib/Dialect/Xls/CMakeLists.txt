set_property(GLOBAL APPEND PROPERTY DYNAMATIC_DIALECT_LIBS XlsDialect)

add_mlir_library(XlsDialect
  ${DYNAMATIC_XLS_DIR}/xls/contrib/mlir/IR/xls_ops.cc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
  MLIRAffineDialect

  DEPENDS
  MLIRXlsIncGen
  MLIRXlsInterfaceIncGen
  dynamatic-headers
)

# XLS is written in C++20, e.g., it relies on the compiler deriving `!=` from
# `==`
set_target_properties(obj.XlsDialect PROPERTIES CXX_STANDARD 20)

# XLS sources feature include paths relative to `llvm-project`, not relative
# to `llvm-project/llvm/include` and `llvm-project/mlir/include`.
# I am so sorry...
foreach(LLVM_INCLUDE_DIR IN LISTS LLVM_INCLUDE_DIRS)
  if(EXISTS ${LLVM_INCLUDE_DIR}/../../llvm/include)
    target_include_directories(XlsDialect PUBLIC ${LLVM_INCLUDE_DIR}/../..)
  else()
    # The prebuilt LLVM installs the headers of all projects into one include
    # directory, so link it as `llvm/include` and `mlir/include` of a directory
    # that mimics the layout of `llvm-project`
    set(LLVM_PROJECT_LAYOUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-project-layout)
    foreach(PROJECT llvm mlir)
      file(MAKE_DIRECTORY ${LLVM_PROJECT_LAYOUT_DIR}/${PROJECT})
      file(CREATE_LINK ${LLVM_INCLUDE_DIR}
           ${LLVM_PROJECT_LAYOUT_DIR}/${PROJECT}/include SYMBOLIC)
    endforeach()
    target_include_directories(XlsDialect PUBLIC ${LLVM_PROJECT_LAYOUT_DIR})
  endif()
endforeach()
