#include <gtest/gtest.h>

#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TypeSystem.h"

using namespace dynamatic;

TEST(TypeSystemTests, BinOpParamOnly) {
  // Bool representing whether a parameter is required.
  class PlusOfTwoParamOnlyTypeSystem
      : public gen::TypeSystem</*TypingContext=*/bool,
                               PlusOfTwoParamOnlyTypeSystem> {
  public:
    static std::optional<ConclusionOf<ast::BinaryExpression>>
    checkBinaryExpression(ast::BinaryExpression::Op op, bool mustBeParameter) {
      // Saw a binop, parameter is now required.
      if (!mustBeParameter && op == ast::BinaryExpression::Plus)
        return ConclusionOf<ast::BinaryExpression>{true, true};

      return std::nullopt;
    }

    static std::optional<ConclusionOf<ast::Parameter>>
    checkParameter(const ast::Parameter &, bool mustBeParameter) {
      if (!mustBeParameter)
        return std::nullopt;

      return mustBeParameter;
    }

    static std::optional<ConclusionOf<ast::Variable>>
    checkVariable(bool mustBeParameter) {
      if (mustBeParameter)
        return mustBeParameter;
      return std::nullopt;
    }

    static std::optional<ConclusionOf<ast::ScalarType>>
    checkScalarType(const ast::ScalarType &scalarType, bool) {
      if (scalarType != ast::PrimitiveType::Double)
        return std::nullopt;

      return ConclusionOf<ast::ScalarType>{};
    }

    static std::optional<ConclusionOf<ast::Constant>>
    checkConstant(const ast::Constant &, bool) {
      return std::nullopt;
    }

    static std::optional<ConclusionOf<ast::CastExpression>>
    checkCastExpression(bool) {
      return std::nullopt;
    }

    static std::optional<ConclusionOf<ast::ConditionalExpression>>
    checkConditionalExpression(bool) {
      return std::nullopt;
    }
  };

  Randomly randomly(/*seed=*/42);
  PlusOfTwoParamOnlyTypeSystem typeSystem;
  gen::BasicCGenerator generator(randomly, typeSystem, /*entryContext=*/false);
  std::string s;
  llvm::raw_string_ostream os(s);
  os << generator.generate("test");

  ASSERT_EQ(s, R"(double test(double var0) {
  return (var0 + var0);
}
)");
}
