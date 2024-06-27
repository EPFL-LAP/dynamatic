add_dynamatic_library(DynamaticSupport
  Attribute.cpp  
  Backedge.cpp
  CFG.cpp
  DOTPrinter.cpp
  Logging.cpp
  MILP.cpp
  RTL.cpp
  System.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  DynamaticHW
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)
