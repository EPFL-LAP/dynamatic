add_dynamatic_library(DynamaticSupport
  DOTPrinter.cpp
  Logging.cpp
  LogicBB.cpp
  NameUniquer.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
