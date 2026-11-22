# Generate the C++ code of the dialect from Banba.td
add_mlir_dialect(Banba banba)

# Generate the C++ code of the dialect's attributes
set(LLVM_TARGET_DEFINITIONS Banba.td)
mlir_tablegen(BanbaAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=banba)
mlir_tablegen(BanbaAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=banba)
add_public_tablegen_target(MLIRBanbaAttrsIncGen)

# Generate the C++ code of the dialect's interfaces
set(LLVM_TARGET_DEFINITIONS BanbaInterfaces.td)
mlir_tablegen(BanbaInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(BanbaInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRBanbaInterfacesIncGen)

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(lsq-generator-banba
  lsq-generator-banba.cpp
  BanbaDialect.cpp
  LsqConfig.cpp

  DEPENDS
  MLIRBanbaIncGen
  MLIRBanbaAttrsIncGen
  MLIRBanbaInterfacesIncGen
)

llvm_update_compile_flags(lsq-generator-banba)
target_include_directories(lsq-generator-banba PRIVATE
  ${CIRCT_INCLUDE_DIRS}
  # The generated files are written to the build folder
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(lsq-generator-banba
  PRIVATE
  CIRCTHW
  CIRCTSeq

  MLIRIR
  MLIRSupport
  )
