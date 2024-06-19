add_dynamatic_library(DynamaticSupport
  Attribute.cpp  
  Backedge.cpp
  CFG.cpp
  DOTPrinter.cpp
  Logging.cpp
  MILP.cpp
  RTL.cpp
  TimingModels.cpp
  Lexer.cpp
  Parser.cpp
  BooleanExpression.cpp
  BooleanLogicLibrary.cpp

  LINK_LIBS PUBLIC
  DynamaticHW
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)
