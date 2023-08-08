add_dynamatic_library(DynamaticSupport
  BufferingStrategy.cpp
  DOTPrinter.cpp
  LogicBB.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
