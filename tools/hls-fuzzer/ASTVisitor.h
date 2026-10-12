#ifndef DYNAMATIC_HLS_FUZZER_ASTVISITOR
#define DYNAMATIC_HLS_FUZZER_ASTVISITOR

#include "AST.h"

#include "llvm/ADT/TypeSwitch.h"

namespace dynamatic {

/// CRTP base class for traversing the C AST.
/// See the file comment for usage. The 'Self' template type parameter should be
/// the class deriving from 'ASTVisitor'.
template <typename Self>
class ASTVisitor {
public:
  /// Shorthand for derived classes to be able to call the default
  /// implementation of methods, e.g. to continue the recursive traversal into
  /// a node's children.
  using Super = ASTVisitor;

  void visit(const ast::Function &function) {
    self().visit(function.returnType);
    for (const ast::ScalarParameter &param : function.scalarParameters)
      self().visit(param);
    for (const ast::ArrayParameter &param : function.arrayParameters)
      self().visit(param);
    for (const ast::Statement &statement : function.statements)
      self().visit(statement);
    if (function.returnStatement)
      self().visit(*function.returnStatement);
  }

  void visit(const ast::Expression &expression) {
    llvm::TypeSwitch<ast::Expression>(expression)
        .Case([&](const ast::Constant *node) { self().visit(*node); })
        .Case([&](const ast::Variable *node) { self().visit(*node); })
        .Case([&](const ast::BinaryExpression *node) { self().visit(*node); })
        .Case([&](const ast::CastExpression *node) { self().visit(*node); })
        .Case([&](const ast::UnaryExpression *node) { self().visit(*node); })
        .Case([&](const ast::ConditionalExpression *node) {
          self().visit(*node);
        })
        .Case(
            [&](const ast::ArrayReadExpression *node) { self().visit(*node); });
  }

  void visit(const ast::Constant &constant) {
    // Leaf node, nothing to recurse into.
  }

  void visit(const ast::Variable &variable) {
    self().visit(variable.getType());
  }

  void visit(const ast::BinaryExpression &expression) {
    self().visit(expression.getLhs());
    self().visit(expression.getRhs());
  }

  void visit(const ast::CastExpression &expression) {
    self().visit(expression.getType());
    self().visit(expression.getExpression());
  }

  void visit(const ast::UnaryExpression &expression) {
    self().visit(expression.getExpression());
  }

  void visit(const ast::ConditionalExpression &expression) {
    self().visit(expression.getCondition());
    self().visit(expression.getTrueVal());
    self().visit(expression.getFalseVal());
  }

  void visit(const ast::ArrayReadExpression &expression) {
    self().visit(expression.getType());
    self().visit(expression.getIndex());
  }

  void visit(const ast::Statement &statement) {
    llvm::TypeSwitch<ast::Statement>(statement)
        .Case([&](const ast::ArrayAssignmentStatement *node) {
          self().visit(*node);
        })
        .Case([&](const ast::StructuredForStatement *node) {
          self().visit(*node);
        });
  }

  void visit(const ast::ReturnStatement &statement) {
    self().visit(statement.getReturnValue());
  }

  void visit(const ast::ArrayAssignmentStatement &statement) {
    self().visit(statement.getIndexingExpression());
    self().visit(statement.getValueExpression());
  }

  void visit(const ast::StructuredForStatement &statement) {
    self().visit(statement.getStart());
    self().visit(statement.getEnd());
    self().visit(statement.getStep());
    for (const ast::Statement &child : statement.getStatements())
      self().visit(child);
  }

  void visit(const ast::ScalarParameter &parameter) {
    self().visit(parameter.getDataType());
  }

  void visit(const ast::ArrayParameter &parameter) {
    self().visit(parameter.getElementType());
  }

  void visit(const ast::ScalarType &type) {
    llvm::TypeSwitch<ast::ScalarType>(type).Case(
        [&](const ast::PrimitiveType *node) { self().visit(*node); });
  }

  void visit(const ast::PrimitiveType &type) {
    // Leaf node, nothing to recurse into.
  }

  void visit(const ast::ReturnType &type) {
    llvm::TypeSwitch<ast::ReturnType>(type)
        .Case([&](const ast::VoidType *node) { self().visit(*node); })
        .Case([&](const ast::ScalarType *node) { self().visit(*node); });
  }

  void visit(const ast::VoidType &type) {
    // Leaf node, nothing to recurse into.
  }

private:
  Self &self() { return static_cast<Self &>(*this); }

  const Self &self() const { return static_cast<const Self &>(*this); }
};

} // namespace dynamatic

#endif
