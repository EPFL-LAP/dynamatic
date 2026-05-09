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

enum class PlusOfTwoState {
  PlusNeeded,
  FreshParamNeeded,
  ExistingParamNeeded,
};

class PlusOfTwoParamOnlyTypeSystem final
    : public gen::DisallowByDefaultTypeSystem<PlusOfTwoState,
                                              PlusOfTwoParamOnlyTypeSystem> {
public:
  using DisallowByDefaultTypeSystem::DisallowByDefaultTypeSystem;

  static bool discardBinaryExpression(ast::BinaryExpression::Op op,
                                      PlusOfTwoState state) {
    return op != ast::BinaryExpression::Plus ||
           state != PlusOfTwoState::PlusNeeded;
  }

  gen::TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op) override {
    return {
        /*lhs=*/TransferFn<ast::BinaryExpression>(
            PlusOfTwoState::FreshParamNeeded),
        /*rhs=*/
        TransferFn<ast::BinaryExpression, ast::BinaryExpression::LHS>(
            PlusOfTwoState::ExistingParamNeeded),
        /*output=*/copyFromParent<ast::BinaryExpression>(),
    };
  }

  static bool discardFreshScalarParameter(PlusOfTwoState state) {
    return state != PlusOfTwoState::FreshParamNeeded;
  }

  static bool discardExistingScalarParameter(const ast::ScalarParameter &,
                                             PlusOfTwoState state) {
    return false;
  }

  static bool discardVariable(PlusOfTwoState state) {
    return state == PlusOfTwoState::PlusNeeded;
  }

  static bool discardScalarType(const ast::ScalarType &scalarType,
                                PlusOfTwoState) {
    return scalarType != ast::PrimitiveType::Double;
  }

  bool discardReturnType(const ast::ReturnType &returnType,
                         PlusOfTwoState state) {
    if (llvm::isa<ast::VoidType>(returnType))
      return true;

    return TypeSystem::discardReturnType(returnType, state);
  }

  constexpr static std::string_view result =
      R"(double test(double var0) {
  return (var0 + var0);
}
)";

  constexpr static auto entryContext = PlusOfTwoState::PlusNeeded;
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

  gen::TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return {
        /*array parameter=*/TransferFn<ast::ArrayReadExpression>(false),
        /*index=*/TransferFn<ast::ArrayReadExpression>(false),
        /*output=*/copyFromParent<ast::ArrayReadExpression>(),
    };
  }

  static bool discardFreshArrayParameter(bool createArrayRead) { return false; }

  static bool discardScalarType(const ast::ScalarType &scalarType,
                                bool /*createArrayRead*/) {
    return scalarType != ast::PrimitiveType::Double;
  }

  bool discardReturnType(const ast::ReturnType &returnType, bool state) {
    return TypeSystem::discardReturnType(returnType, state);
  }

  static std::optional<ast::Constant> discardConstant(const ast::Constant &,
                                                      bool createArrayRead) {
    if (createArrayRead)
      return std::nullopt;

    return ast::Constant{0};
  }

  constexpr static std::string_view result =
      R"(double test(double var0[32]) {
  return var0[((uint32_t)((0)) & (31u))];
}
)";

  constexpr static auto entryContext = true;
};

} // namespace

using MyTypes = ::testing::Types<PlusOfTwoParamOnlyTypeSystem,
                                 ReturnArrayConstantOnlyTypeSystem>;
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
INSTANTIATE_TYPED_TEST_SUITE_P(All, TypeSystemTest, MyTypes);
