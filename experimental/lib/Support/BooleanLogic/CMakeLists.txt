add_dynamatic_library(DynamaticExperimentalSupportBooleanLogic
  BoolExpression.cpp
  Lexer.cpp
  Parser.cpp
  ROBDD.cpp

  LINK_LIBS PRIVATE
  DynamaticSupportEspresso
)
