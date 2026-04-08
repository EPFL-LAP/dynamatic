#include <gtest/gtest.h>

#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TypeSystem.h"

using namespace dynamatic;

template <typename TypeSystem>
class TypeSystemTest : public testing::Test {};

TYPED_TEST_SUITE_P(TypeSystemTest);

TYPED_TEST_P(TypeSystemTest, OutputCheck) {
  Randomly randomly(/*seed=*/42);
  TypeParam typeSystem(randomly);
  gen::BasicCGenerator generator(randomly, typeSystem,
                                 /*entryContext=*/typeSystem.entryContext);
  std::string s;
  llvm::raw_string_ostream os(s);
  os << generator.generate("test");

  ASSERT_EQ(s, typeSystem.result);
}

REGISTER_TYPED_TEST_SUITE_P(TypeSystemTest, OutputCheck);

namespace {
// Bool representing whether a parameter is required.
class PlusOfTwoParamOnlyTypeSystem final
    : public gen::DisallowByDefaultTypeSystem<bool,
                                              PlusOfTwoParamOnlyTypeSystem> {
public:
  using DisallowByDefaultTypeSystem::DisallowByDefaultTypeSystem;

  static std::optional<ConclusionOf<ast::BinaryExpression>>
  checkBinaryExpression(ast::BinaryExpression::Op op, bool mustBeParameter) {
    // Saw a binop, parameter is now required.
    if (!mustBeParameter && op == ast::BinaryExpression::Plus)
      return ConclusionOf<ast::BinaryExpression>{true, true};

    return std::nullopt;
  }

  static std::optional<ConclusionOf<ast::ScalarParameter>>
  checkScalarParameter(const ast::ScalarParameter &, bool mustBeParameter) {
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

  constexpr static std::string_view result =
      R"(double test(double var0) {
  return (var0 + var0);
}
)";

  constexpr static auto entryContext = false;
};

// Bool representing whether an array read expression is required.
// Otherwise, a 0 constant must be generated.
class ReturnArrayConstantOnlyTypeSystem final
    : public gen::DisallowByDefaultTypeSystem<
          /*createArrayRead=*/bool, ReturnArrayConstantOnlyTypeSystem> {
public:
  using DisallowByDefaultTypeSystem::DisallowByDefaultTypeSystem;

  static std::optional<ConclusionOf<ast::ArrayReadExpression>>
  checkArrayReadExpression(bool createArrayRead) {
    if (!createArrayRead)
      return std::nullopt;
    return ConclusionOf<ast::ArrayReadExpression>{false, false};
  }

  static std::optional<ast::ArrayParameter> generateFreshArrayParameter(
      bool context, GenerateCallback<ast::ScalarType, bool> generateScalarType,
      llvm::function_ref<std::string()> generateFreshVarName) {
    std::optional<ast::ScalarType> elementType = generateScalarType(context);
    if (!elementType)
      return std::nullopt;

    return ast::ArrayParameter{
        std::move(*elementType),
        generateFreshVarName(),
        /*dimension=*/32,
    };
  }

  static std::optional<ConclusionOf<ast::ScalarType>>
  checkScalarType(const ast::ScalarType &scalarType, bool /*createArrayRead*/) {
    if (scalarType != ast::PrimitiveType::Double)
      return std::nullopt;

    return ConclusionOf<ast::ScalarType>{};
  }

  static std::optional<ConclusionOf<ast::Constant>>
  checkConstant(const ast::Constant &, bool createArrayRead) {
    if (createArrayRead)
      return std::nullopt;

    return ast::Constant{0};
  }

  constexpr static std::string_view result =
      R"(double test(double var0[32]) {
  return var0[((uint32_t)(0) & 31u)];
}
)";

  constexpr static auto entryContext = true;
};

} // namespace

using MyTypes = ::testing::Types<PlusOfTwoParamOnlyTypeSystem,
                                 ReturnArrayConstantOnlyTypeSystem>;
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
INSTANTIATE_TYPED_TEST_SUITE_P(All, TypeSystemTest, MyTypes);
