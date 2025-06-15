add_dynamatic_library(DynamaticExperimentalSupportBooleanLogic
  BoolExpression.cpp
  Lexer.cpp
  Parser.cpp
  BDD.cpp

  LINK_LIBS PRIVATE
  DynamaticSupportEspresso
)
