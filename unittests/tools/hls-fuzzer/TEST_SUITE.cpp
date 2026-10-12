#include <gtest/gtest.h>

#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/OptionalTypeSystem.h"
#include "hls-fuzzer/TemplateTypeSystem.h"
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
  os << generator.generateFunction("test");

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
        /*output=*/copyInputToOutput<ast::BinaryExpression>(),
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
        /*output=*/copyInputToOutput<ast::ArrayReadExpression>(),
    };
  }

  static bool discardFreshArrayParameter(bool createArrayRead) { return false; }

  static bool discardScalarType(const ast::ScalarType &scalarType,
                                bool /*createArrayRead*/) {
    return scalarType != ast::PrimitiveType::Double;
  }

  bool discardReturnType(const ast::ReturnType &returnType, bool state) {
    if (returnType == ast::VoidType{})
      return true;

    return TypeSystem::discardReturnType(returnType, state);
  }

  static std::optional<ast::Constant> discardConstant(const ast::Constant &,
                                                      bool createArrayRead) {
    if (createArrayRead)
      return std::nullopt;

    return ast::Constant{0};
  }

  static std::optional<std::size_t> discardArrayDimension(std::size_t, bool) {
    return 8;
  }

  constexpr static std::string_view result =
      R"(double test(double var0[8]) {
  return var0[((uint32_t)((0)) & (7u))];
}
)";

  constexpr static auto entryContext = true;
};

/// Context of 'ForwardStatementsTypeSystem'.
struct ForwardStatementsState {
  /// What the generator is currently being asked to produce. Everything not
  /// mentioned here is generated in the 'Statement' state.
  enum Kind {
    /// A statement of the function's body.
    Statement,
    /// The variable an assignment statement writes to.
    AssignmentTarget,
    /// The value an assignment statement writes.
    AssignmentValue,
    /// The expression returned by the function.
    ReturnValue,
  } kind = Statement;

  /// Number of statements generated before the one currently being generated.
  /// It doubles as the constant that statement assigns.
  std::size_t statementIndex = 0;
};

/// Type system generating one completely fixed program:
///
///   void test(int32_t var0) {
///     var0 = (0);
///     var0 = (1);
///     var0 = (2);
///     var0 = (3);
///   }
///
/// Its purpose is to pin down that statement lists are generated in forward
/// direction: 'ast::StatementList' is right recursive, so the head statement is
/// generated first and the rest of the list afterwards. This type system relies
/// on that ordering, as the constant a statement assigns is derived from the
/// number of statements generated before it. Were the list generated backwards,
/// the constants would count down instead.
class ForwardStatementsTypeSystem final
    : public gen::DisallowByDefaultTypeSystem<ForwardStatementsState,
                                              ForwardStatementsTypeSystem> {
  /// Number of statements the function body consists of.
  constexpr static std::size_t NUM_STATEMENTS = 4;

  /// Type of the single variable the program assigns to.
  constexpr static ast::PrimitiveType::Type VAR_TYPE =
      ast::PrimitiveType::Int32;

public:
  using DisallowByDefaultTypeSystem::DisallowByDefaultTypeSystem;

  static bool discardReturnType(const ast::ReturnType &returnType,
                                ForwardStatementsState) {
    // The program consists of nothing but its assignments.
    return !llvm::isa<ast::VoidType>(returnType);
  }

  gen::TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    // A return statement is generated even for a void function, in which case
    // the generator drops it again. It still has to be generatable, hence the
    // dedicated state allowing a constant below.
    // TODO: This is a bug in the generator. Ideally we should have some way
    //       that generating a noop return statement in a void function is not
    //       necessary.
    return {
        /*return value=*/
        TransferFn<ast::ReturnStatement, gen::INPUT_DEPENDENCY>(
            [](ForwardStatementsState state) {
              state.kind = ForwardStatementsState::ReturnValue;
              return state;
            }),
        /*output=*/copyInputToOutput<ast::ReturnStatement>(),
    };
  }

  static bool discardStatementList(ForwardStatementsState state) {
    return state.statementIndex >= NUM_STATEMENTS;
  }

  gen::TransferFnArray<ast::StatementList>
  getStatementListTransferFns() override {
    return {
        // The head is generated from the list's input state, and the rest of
        // the list from the head's output state, which counted the head. The
        // dependency on STATEMENT is also what forces the head to be generated
        // first and hence what lets 'discardStatementList' terminate the
        // recursion.
        /*statement=*/copyFromInput<ast::StatementList>(),
        /*statement list=*/
        copyFrom<ast::StatementList, ast::StatementList::STATEMENT>(),
        /*output=*/
        copyToOutput<ast::StatementList, ast::StatementList::STATEMENT_LIST>(),
    };
  }

  static bool discardScalarAssignmentStatement(ForwardStatementsState state) {
    return state.kind != ForwardStatementsState::Statement;
  }

  gen::TransferFnArray<ast::ScalarAssignmentStatement>
  getScalarAssignmentStatementTransferFns() override {
    return {
        /*target=*/
        TransferFn<ast::ScalarAssignmentStatement, gen::INPUT_DEPENDENCY>(
            [](ForwardStatementsState state) {
              state.kind = ForwardStatementsState::AssignmentTarget;
              return state;
            }),
        /*value=*/
        TransferFn<ast::ScalarAssignmentStatement, gen::INPUT_DEPENDENCY,
                   ast::ScalarAssignmentStatement::TARGET>(
            [](ForwardStatementsState state, const ForwardStatementsState &,
               const ast::ScalarParameter &) {
              // Depending on the target forces it to be generated first, so
              // that the very first statement is the one creating the variable.
              state.kind = ForwardStatementsState::AssignmentValue;
              return state;
            }),
        /*output=*/
        OutputTransferFn<ast::ScalarAssignmentStatement, gen::INPUT_DEPENDENCY>(
            [](const ast::ScalarAssignmentStatement &,
               ForwardStatementsState state) {
              ++state.statementIndex;
              return state;
            }),
    };
  }

  static bool discardFreshScalarParameter(ForwardStatementsState state) {
    // Only the first statement creates the variable; every later one reuses it.
    return state.kind != ForwardStatementsState::AssignmentTarget ||
           state.statementIndex != 0;
  }

  static bool discardExistingScalarParameter(const ast::ScalarParameter &,
                                             ForwardStatementsState state) {
    return state.kind != ForwardStatementsState::AssignmentTarget ||
           state.statementIndex == 0;
  }

  static bool discardScalarType(const ast::ScalarType &scalarType,
                                ForwardStatementsState) {
    return scalarType != VAR_TYPE;
  }

  static std::optional<ast::Constant>
  discardConstant(const ast::Constant &, ForwardStatementsState state) {
    // 'discardConstant' may replace the randomly drawn constant with a fixed
    // one, which is how the assigned values are pinned down.
    switch (state.kind) {
    case ForwardStatementsState::AssignmentValue:
      return ast::Constant{static_cast<std::int32_t>(state.statementIndex)};
    case ForwardStatementsState::ReturnValue:
      // Discarded together with the return statement of the void function.
      return ast::Constant{std::int32_t{0}};
    default:
      return std::nullopt;
    }
  }

  constexpr static std::string_view result =
      R"(void test(int32_t var0) {
  var0 = (0);
  var0 = (1);
  var0 = (2);
  var0 = (3);
}
)";

  constexpr static auto entryContext = ForwardStatementsState{};
};

/// Type system whose typing context is the value that a generated constant is
/// required to have. Constants are the only AST node it allows.
///
/// It is therefore only usable in conjunction with a type system generating the
/// rest of the program, such as 'PlusExpressionTypeSystem'.
class SpecificConstantTypeSystem final
    : public gen::DisallowByDefaultTypeSystem</*requiredValue=*/std::uint32_t,
                                              SpecificConstantTypeSystem> {
public:
  using DisallowByDefaultTypeSystem::DisallowByDefaultTypeSystem;

  /// Replaces whatever constant the generator suggested with the one required
  /// by the context.
  static std::optional<ast::Constant>
  discardConstant(const ast::Constant &, std::uint32_t requiredValue) {
    return ast::Constant{requiredValue};
  }
};

// State denoting whether a plus expression still has to be generated or
// whether one of its operands is currently being generated.
enum class PlusExpressionState {
  PlusNeeded,
  OperandNeeded,
};

/// Type system generating a single 'uint32_t' plus expression whose operands
/// are constants.
///
/// Note that this type system does not constrain the value of the operands. It
/// is meant to be used in conjunction with a type system doing so, such as
/// 'SpecificConstantTypeSystem'.
class PlusExpressionTypeSystem final
    : public gen::DisallowByDefaultTypeSystem<PlusExpressionState,
                                              PlusExpressionTypeSystem> {
public:
  using DisallowByDefaultTypeSystem::DisallowByDefaultTypeSystem;

  static bool discardBinaryExpression(ast::BinaryExpression::Op op,
                                      PlusExpressionState state) {
    return op != ast::BinaryExpression::Plus ||
           state != PlusExpressionState::PlusNeeded;
  }

  gen::TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op) override {
    return {
        /*lhs=*/TransferFn<ast::BinaryExpression>(
            PlusExpressionState::OperandNeeded),
        /*rhs=*/
        TransferFn<ast::BinaryExpression>(PlusExpressionState::OperandNeeded),
        /*output=*/copyInputToOutput<ast::BinaryExpression>(),
    };
  }

  /// Constants are only legal as operands of the plus expression. This forces
  /// the plus expression to be generated first.
  static std::optional<ast::Constant>
  discardConstant(const ast::Constant &constant, PlusExpressionState state) {
    if (state == PlusExpressionState::PlusNeeded)
      return std::nullopt;

    return constant;
  }

  static bool discardScalarType(const ast::ScalarType &scalarType,
                                PlusExpressionState) {
    return scalarType != ast::PrimitiveType::UInt32;
  }

  bool discardReturnType(const ast::ReturnType &returnType,
                         PlusExpressionState state) {
    if (llvm::isa<ast::VoidType>(returnType))
      return true;

    return TypeSystem::discardReturnType(returnType, state);
  }

  constexpr static auto entryContext = PlusExpressionState::PlusNeeded;
};

/// Conjunction of 'PlusExpressionTypeSystem' and an optional
/// 'SpecificConstantTypeSystem'.
/// The former decides the shape of the program, the latter the value of the
/// constants within it. The constant type system is disabled outside the
/// operands of the plus expression, where it is enabled by transfer functions
/// of this type system: the left-hand side is required to be '0', the
/// right-hand side that value plus one.
/// The constant type system is disabled by default and only enabled where a
/// constant with a specific value is required.
using OptionalConstantTypeSystem =
    gen::OptionalTypeSystem<SpecificConstantTypeSystem>;

class ZeroPlusOneTypeSystem final
    : public gen::ConjunctionTypeSystemBase<ZeroPlusOneTypeSystem,
                                            OptionalConstantTypeSystem,
                                            PlusExpressionTypeSystem> {

public:
  ZeroPlusOneTypeSystem()
      : ConjunctionTypeSystemBase(SpecificConstantTypeSystem(),
                                  PlusExpressionTypeSystem()) {}

  gen::TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    gen::TransferFnArray<ast::BinaryExpression> transferFns =
        ConjunctionTypeSystemBase::getBinaryExpressionTransferFns(op);

    // Enable the constant type system for both operands: the left-hand side
    // is required to be '0'...
    crossTransferFns<ast::BinaryExpression::LHS, OptionalConstantTypeSystem>(
        transferFns, [] { return 0; });
    // ...while the right-hand side derives its required value from the one of
    // the left-hand side, forcing the latter to be generated first.
    crossTransferFns<
        ast::BinaryExpression::RHS, OptionalConstantTypeSystem,
        Dep<OptionalConstantTypeSystem, ast::BinaryExpression::LHS>,
        // Note: Redundant contexts that are not used, but used to test that it
        // can also depend on the RHS of another typesystem and its own!
        Dep<PlusExpressionTypeSystem, ast::BinaryExpression::RHS>,
        Dep<OptionalConstantTypeSystem, ast::BinaryExpression::RHS>>(
        transferFns,
        [](const OptionalConstantTypeSystem::Context &lhsContext,
           const PlusExpressionTypeSystem::Context &rhsContext,
           const OptionalConstantTypeSystem::Context &rhsContextAgain) {
          (void)rhsContext;
          (void)rhsContextAgain;

          return *lhsContext + 1;
        });

    return transferFns;
  }

  constexpr static std::string_view result =
      R"(uint32_t test() {
  return ((0u) + (1u));
}
)";

  /// The constant type system is initially disabled.
  constexpr static Context entryContext = {
      std::nullopt, PlusExpressionTypeSystem::entryContext};
};

} // namespace

using MyTypes =
    ::testing::Types<PlusOfTwoParamOnlyTypeSystem,
                     ReturnArrayConstantOnlyTypeSystem,
                     ForwardStatementsTypeSystem, ZeroPlusOneTypeSystem>;
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
INSTANTIATE_TYPED_TEST_SUITE_P(All, TypeSystemTest, MyTypes);
