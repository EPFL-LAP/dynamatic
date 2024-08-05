add_dynamatic_library(DynamaticSupport
  Attribute.cpp  
  Backedge.cpp
  CFG.cpp
  DOTPrinter.cpp
  Logging.cpp
  MILP.cpp
  System.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  DynamaticHandshake
  DynamaticHW
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)

add_subdirectory(Espresso)
add_subdirectory(RTL)
