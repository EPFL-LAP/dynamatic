add_dynamatic_library(DynamaticSupport
  DOTPrinter.cpp
  LogicBB.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
