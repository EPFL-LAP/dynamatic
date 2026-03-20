#ifndef DYNAMATIC_HLS_FUZZER_BasicCProducer
#define DYNAMATIC_HLS_FUZZER_BasicCProducer

#include "AST.h"
#include "Randomly.h"
#include "TypeSystem.h"

#include <optional>

namespace dynamatic::gen {

/// Base generator which is responsible for generating valid C programs that do
/// not contain undefined behavior.
///
/// The generated programs can be further restricted through the use of a type
/// system which constrains the kind of valid expressions the generator may
/// output.
///
/// Generating new C constructs should be implemented in this class.
class BasicCGenerator {
public:
  /// Constructs a new base generator which generates programs that adhere to
  /// the given type system.
  /// 'entryContext' is entry state used to type check the function returned by
  /// 'generate'.
  template <class TypingContext, class Self>
  explicit BasicCGenerator(Randomly &random,
                           TypeSystem<TypingContext, Self> &typeSystem,
                           const TypingContext &entryContext = {})
      : random(random), typeSystem(typeSystem), entryContext(entryContext) {}

  /// Returns a new function with the given function name.
  ast::Function generate(std::string_view functionName);

  /// Generates a dynamatic test bench for the given function.
  // TODO: Could return a function once our AST is powerful enough to represent
  // entire test benches.
  std::string generateTestBench(const ast::Function &kernel) const;

private:
  std::string generateFreshVarName() {
    return "var" + std::to_string(varCounter++);
  }

  const ast::Parameter &generateFreshParameter(ast::ScalarType datatype,
                                               const OpaqueContext &context);

  ast::ReturnStatement generateFunctionBody(const OpaqueContext &constraints);

  ast::Expression generateExpression(const OpaqueContext &constraint,
                                     std::size_t depth);

  std::optional<ast::Expression>
  generateBinaryExpression(ast::BinaryExpression::Op op,
                           const OpaqueContext &constraints, std::size_t depth);

  std::optional<ast::ConditionalExpression>
  generateConditionalExpression(const OpaqueContext &constraint,
                                std::size_t depth);

  std::optional<ast::CastExpression>
  generateCastExpression(const OpaqueContext &constraint, std::size_t depth);

  std::optional<ast::Constant> generateConstant(const OpaqueContext &constraint,
                                                std::size_t depth = 0) const;

  std::optional<ast::Variable>
  generateScalarParameter(const OpaqueContext &constraints,
                          std::size_t depth = 0);

  ast::ScalarType generateScalarType(const OpaqueContext &constraints) const;

  Randomly &random;
  ast::ScalarType returnType{};
  std::vector<std::pair<ast::Parameter, OpaqueContext>> parameters;
  std::size_t varCounter = 0;
  AbstractTypeSystem &typeSystem;
  OpaqueContext entryContext;
};

} // namespace dynamatic::gen

#endif
