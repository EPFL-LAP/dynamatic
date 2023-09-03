add_dynamatic_library(DynamaticSupport
  DOTPrinter.cpp
  LogicBB.cpp
  TimingModels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
