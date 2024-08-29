add_dynamatic_library(DynamaticExperimentalConversion
  FtdCfToHandshake.cpp
<<<<<<< HEAD
  FtdMemoryInterface.cpp
=======
>>>>>>> 1762f18 (Skeleton of FTD `cf` to `handshake` conversion pass)

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHandshake
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
<<<<<<< HEAD
  DynamaticExperimentalSupportBooleanLogic
  DynamaticSupportEspresso
  DynamaticExperimentalSupport
=======
>>>>>>> 1762f18 (Skeleton of FTD `cf` to `handshake` conversion pass)
  )

