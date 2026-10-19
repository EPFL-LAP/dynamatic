# lib/dynamatic/Dialect/CFExtra/CMakeLists.txt

add_dynamatic_dialect_library(DynamaticCFExtra
  CFExtraDialect.cpp
  CFExtraOps.cpp

  DEPENDS
  MLIRCFExtraIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)