add_dynamatic_library(DynamaticExperimentalSupportBooleanLogic
  BoolExpression.cpp
  Lexer.cpp
  Parser.cpp
  BDD.cpp
  ReadOnceBDD.cpp

  LINK_LIBS PRIVATE
  DynamaticSupportEspresso
)
