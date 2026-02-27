add_dynamatic_library(DynamaticSupport
  Attribute.cpp
  Backedge.cpp
  BLIFFileManager.cpp
  CFG.cpp
  DOT.cpp
  MILP.cpp
  System.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  DynamaticSupportJSON
  DynamaticHandshake
  DynamaticLinearAlgebra
  DynamaticHW
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)

add_subdirectory(Espresso)
add_subdirectory(JSON)
add_subdirectory(RTL)
add_subdirectory(Utils)
<<<<<<< HEAD
add_subdirectory(ConstraintProgramming)
add_subdirectory(BlifImporter)
add_subdirectory(BlifExporter)
add_subdirectory(LinearAlgebra)
=======
add_subdirectory(ConstraintProgramming)
>>>>>>> 6b2b6b747 (chore: add squashed changes)
