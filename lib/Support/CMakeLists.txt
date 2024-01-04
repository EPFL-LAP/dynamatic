add_dynamatic_library(DynamaticSupport
  CFG.cpp
  DOTPrinter.cpp
  Handshake.cpp
  Logging.cpp
  MILP.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
