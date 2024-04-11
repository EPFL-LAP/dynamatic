add_dynamatic_library(DynamaticSupport
  Backedge.cpp
  CFG.cpp
  DOTPrinter.cpp
  Logging.cpp
  MILP.cpp
  RTL.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)
