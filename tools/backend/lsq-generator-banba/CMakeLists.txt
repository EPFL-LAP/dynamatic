# The tablegen rules below only watch the .td files directly in this folder.
# This makes them watch the ones in the sub-folders too
file(GLOB_RECURSE LLVM_TARGET_DEPENDS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.td)

# Banba's own tablegen tool, which generates code that mlir-tblgen does not
add_subdirectory(banba-tblgen)

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

# Generate the named drivers of the dialect's operations, with banba-tblgen
set(LLVM_TARGET_DEFINITIONS Banba.td)
tablegen(BANBA BanbaNamedDrivers.h.inc -gen-banba-named-drivers)
add_public_tablegen_target(MLIRBanbaNamedDriversIncGen)

# Generate the port names of the dialect's operations, with banba-tblgen
set(LLVM_TARGET_DEFINITIONS Banba.td)
tablegen(BANBA BanbaPortNames.cpp.inc -gen-banba-port-names)
add_public_tablegen_target(MLIRBanbaPortNamesIncGen)

# Generate the result names of the dialect's operations, with banba-tblgen
set(LLVM_TARGET_DEFINITIONS Banba.td)
tablegen(BANBA BanbaResultNames.cpp.inc -gen-banba-result-names)
add_public_tablegen_target(MLIRBanbaResultNamesIncGen)

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(lsq-generator-banba
  lsq-generator-banba.cpp
  BanbaDialect.cpp
  common/ConstructCallerParamOps.cpp
  common/ConstructLsq.cpp
  LowerToHw.cpp
  LsqConfig.cpp
  per-port-queues/ConstructGroupAllocator.cpp
  shared-queues/ConstructGroupAllocator.cpp

  DEPENDS
  MLIRBanbaIncGen
  MLIRBanbaAttrsIncGen
  MLIRBanbaInterfacesIncGen
  MLIRBanbaNamedDriversIncGen
  MLIRBanbaPortNamesIncGen
  MLIRBanbaResultNamesIncGen
)

llvm_update_compile_flags(lsq-generator-banba)
target_include_directories(lsq-generator-banba PRIVATE
  ${CIRCT_INCLUDE_DIRS}
  # So that files in the sub-folders find the headers in this folder
  ${CMAKE_CURRENT_SOURCE_DIR}
  # The generated files are written to the build folder
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(lsq-generator-banba
  PRIVATE
  CIRCTComb
  CIRCTExportVerilog
  CIRCTHW
  CIRCTSV
  CIRCTSeq
  CIRCTSeqToSV

  MLIRIR
  MLIRPass
  MLIRSupport
  )
