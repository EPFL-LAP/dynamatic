add_dynamatic_library(DynamaticSupport
  DOTPrinter.cpp
  Logging.cpp
  LogicBB.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
