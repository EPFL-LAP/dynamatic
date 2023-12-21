add_dynamatic_library(DynamaticSupport
  DOTPrinter.cpp
  Handshake.cpp
  Logging.cpp
  LogicBB.cpp
  MILP.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
