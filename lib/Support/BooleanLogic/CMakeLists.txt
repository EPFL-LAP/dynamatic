add_dynamatic_library(DynamaticSupportBooleanLogic
  BoolExpression.cpp
  Lexer.cpp
  Parser.cpp
  BDD.cpp

  LINK_LIBS PRIVATE
  DynamaticSupportEspresso
)
