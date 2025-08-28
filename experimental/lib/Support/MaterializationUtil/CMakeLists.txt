include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Support/MaterializationUtil)

add_dynamatic_library(MaterializationUtil
  MaterializationUtil.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
)
