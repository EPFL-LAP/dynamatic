#include <gtest/gtest.h>

#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TypeSystem.h"

using namespace dynamatic;

template <typename TypeSystem>
class TypeSystemTest : public testing::Test {};

TYPED_TEST_SUITE_P(TypeSystemTest);

TYPED_TEST_P(TypeSystemTest, OutputCheck) {
  Randomly randomly(/*seed=*/42);
  TypeParam typeSystem;
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

  static bool discardBinaryExpression(ast::BinaryExpression::Op op,
                                      bool mustBeParameter) {
    return mustBeParameter || op != ast::BinaryExpression::Plus;
  }

  gen::DependencyArray<ast::BinaryExpression>
  getBinaryExpressionContextDependencies(ast::BinaryExpression::Op) override {
    return {
        Dependency<ast::BinaryExpression>(true),
        Dependency<ast::BinaryExpression>(true),
        Dependency<ast::BinaryExpression>(true),
    };
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

  static bool discardArrayReadExpression(bool createArrayRead) {
    return !createArrayRead;
  }

  gen::DependencyArray<ast::ArrayReadExpression>
  getArrayReadExpressionContextDependencies() override {
    return gen::DependencyArray<ast::ArrayReadExpression>{
        Dependency<ast::ArrayReadExpression>(false),
        Dependency<ast::ArrayReadExpression>(false),
        copyFromParent<ast::ArrayReadExpression>(),
    };
  }

  std::optional<ConclusionOf<ast::ArrayParameter>>
  checkArrayParameter(const ast::ArrayParameter &param, bool createArrayRead) {
    // TODO: The array dimension is currently random making the test below
    //       susceptible to internal implementation changes.
    //       Array parameters (like constants) are terminators with a large
    //       combination of possible values.
    //       We probably want to allow the type system to return an array
    //       parameter to use instead for that reason.
    return TypeSystem::checkArrayParameter(param, createArrayRead);
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
      R"(double test(double var0[1]) {
  return var0[((uint32_t)((0)) & (0u))];
}
)";

  constexpr static auto entryContext = true;
};

} // namespace

using MyTypes = ::testing::Types<PlusOfTwoParamOnlyTypeSystem,
                                 ReturnArrayConstantOnlyTypeSystem>;
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
INSTANTIATE_TYPED_TEST_SUITE_P(All, TypeSystemTest, MyTypes);
