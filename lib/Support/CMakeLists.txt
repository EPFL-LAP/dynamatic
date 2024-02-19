add_dynamatic_library(DynamaticSupport
  Backedge.cpp
  CFG.cpp
  DOTPrinter.cpp
  Handshake.cpp
  Logging.cpp
  MILP.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)
