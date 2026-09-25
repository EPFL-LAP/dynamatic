#ifndef HLS_FUZZER_TARGETS_DISABLELATENCYOPTTYPESYSTEM
#define HLS_FUZZER_TARGETS_DISABLELATENCYOPTTYPESYSTEM

#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"

#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

namespace dynamatic::gen {

/// Implementation details of 'DisableLatencyOptTypeSystem' that are not part of
/// its public interface.
namespace detail {

/// Typing context of 'MulStrengthReductionTypeSystem'.
struct MulOperandTypingContext {
  /// True iff the AST node about to be generated is an operand of a
  /// multiplication, or a cast on one. Note that this is about the operand
  /// itself rather than about the expression tree below it: everything else
  /// clears the flag again.
  bool isMulOperand = false;
};

/// Sub type system that makes '--arith-reduce-strength' less likely to replace
/// a generated integer multiplication with a tree of shifts and adds.
///
/// Its 'MulReduceStrength' pattern rewrites 'x * C', for a strictly positive
/// constant 'C', into the sum of 'x' shifted left by every set bit position of
/// 'C'. It only does so as long as the resulting adder tree stays shallow,
/// i.e. as long as
///
///   ceil(log2(popcount(C))) <= max-adder-depth-mul
///
/// which, for the depth of 3 used by default, is every constant with at most 8
/// set bits: *every* constant representable in 8 bits.
/// This almost certainly changes the latency of a multiply operation
/// considerably which is undesirable in some targets.
///
/// The type system therefore rejects such a constant as the operand of a
/// multiplication.
/// Note that the strength reduction may still trigger after constant folding
/// which this pattern purposefully does not handle.
class MulStrengthReductionTypeSystem final
    : public TypeSystem<MulOperandTypingContext,
                        MulStrengthReductionTypeSystem> {
public:
  /// Maximum adder tree depth used by dynamatic.
  static constexpr unsigned MAX_ADDER_DEPTH_MUL = 3;

  /// Returns true if 'MulReduceStrength' would replace a multiplication by
  /// 'constant' with a tree of shifts and adds.
  static bool isStrengthReducibleMultiplier(const ast::Constant &constant) {
    return std::visit(
        [](auto &&value) {
          using T = std::decay_t<decltype(value)>;
          // Only 'arith::MulIOp' is matched by the pattern, floating point
          // multiplications are left alone.
          if constexpr (!std::is_integral_v<T>) {
            return false;
          } else {
            // Neither is a negative constant ('getPosConstantOperand' requires
            // a strictly positive one) nor a zero, which the canonicalizer
            // folds away long before the pattern gets to see it.
            if (value <= 0)
              return false;

            int setBits = llvm::popcount(static_cast<std::uint64_t>(value));
            return llvm::Log2_64_Ceil(setBits) <= MAX_ADDER_DEPTH_MUL;
          }
        },
        constant.value);
  }

  std::optional<ast::Constant>
  discardConstant(const ast::Constant &constant,
                  const MulOperandTypingContext &context) {
    if (context.isMulOperand && isStrengthReducibleMultiplier(constant))
      return std::nullopt;

    return Super::discardConstant(constant, context);
  }

  /// Every subelement clears the flag by default: only the operands of a
  /// multiplication are banned constants, and any node in between a
  /// multiplication and a constant further down is no longer such an operand.
  template <typename ASTNode>
  static auto defaultTransferFn() {
    return mulOperand<ASTNode>(false);
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    if (op != ast::BinaryExpression::Mul)
      return Super::getBinaryExpressionTransferFns(op);

    // Note: Technically we don't know yet whether this is an integer multiply
    //       (which is the only pattern we want to prevent). We deduce this
    //       from the type of the constant.
    //       TODO: This logic could remove float multiplies using casts without
    //       the dynamatic type system.
    return {
        /*lhs=*/mulOperand<ast::BinaryExpression>(true),
        /*rhs=*/mulOperand<ast::BinaryExpression>(true),
        /*output=*/copyInputToOutput<ast::BinaryExpression>(),
    };
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return {
        /*target type=*/mulOperand<ast::CastExpression>(false),
        // A cast of a constant is folded to a constant before the pass runs, so
        // the operand of one is as much the multiplication's operand as the
        // cast itself is.
        /*operand=*/copyFromInput<ast::CastExpression>(),
        /*output=*/copyInputToOutput<ast::CastExpression>(),
    };
  }

private:
  /// Returns a transfer function stating that a subelement is, or is not, the
  /// operand of a multiplication.
  template <typename ASTNode>
  static TransferFn<ASTNode, INPUT_DEPENDENCY> mulOperand(bool isMulOperand) {
    return TransferFn<ASTNode, INPUT_DEPENDENCY>(
        [isMulOperand](MulOperandTypingContext context) {
          context.isMulOperand = isMulOperand;
          return context;
        });
  }
};

/// Typing context of 'MulSelectFoldTypeSystem'.
struct BooleanMulOperandTypingContext {
  /// True iff the value of the AST node about to be generated is the value of
  /// an operand of a multiplication, i.e. iff the node is such an operand or
  /// sits below one with nothing but 0/1-preserving operators in between (see
  /// 'MulSelectFoldTypeSystem').
  bool isMulOperandValue = false;
};

/// Sub type system that keeps LLVM's instruction combiner from replacing a
/// generated multiplication with a select.
///
/// Its fold rewrites 'x * zext(c)', for a one-bit 'c', into 'c ? x : 0': a
/// multiplication one of whose operands is zero or one is a mask, and a mask is
/// a select. The cycles the multiplier would have taken turn into the zero a
/// select costs, leaving the operation with almost zero latency.
/// This is problematic for some targets where latency
///
/// C spells one-bit values as 'int' but produces them all the same: every
/// comparison and every '!' yields zero or one and is emitted as an 'i1' the
/// operand's type extends.
///
/// A comparison and a '!' are therefore rejected as the operand of a
/// multiplication, as is anything reached from such an operand through the
/// operators that hand a 0/1 value on unchanged: a cast (which merely widens
/// it), the two values of a conditional expression (one of which the result
/// is), and the operands of a bitwise and, or, or xor (whose result is 0/1
/// whenever both of them are).
class MulSelectFoldTypeSystem final
    : public TypeSystem<BooleanMulOperandTypingContext,
                        MulSelectFoldTypeSystem> {
public:
  /// Returns true if 'op' yields zero or one, i.e. whether it is one of the
  /// comparisons.
  static bool isComparison(ast::BinaryExpression::Op op) {
    switch (op) {
    case ast::BinaryExpression::Greater:
    case ast::BinaryExpression::GreaterEqual:
    case ast::BinaryExpression::Less:
    case ast::BinaryExpression::LessEqual:
    case ast::BinaryExpression::Equal:
    case ast::BinaryExpression::NotEqual:
      return true;
    default:
      return false;
    }
  }

  static bool
  discardBinaryExpression(ast::BinaryExpression::Op op,
                          const BooleanMulOperandTypingContext &context) {
    return context.isMulOperandValue && isComparison(op);
  }

  static bool
  discardUnaryExpression(ast::UnaryExpression::Op op,
                         const BooleanMulOperandTypingContext &context) {
    return context.isMulOperandValue && op == ast::UnaryExpression::BoolNot;
  }

  /// Every subelement clears the flag by default: only a multiplication's
  /// operands set it, and only the 0/1-preserving subelements singled out in
  /// the overrides below hand it on.
  template <typename ASTNode>
  static auto defaultTransferFn() {
    return mulOperandValue<ASTNode>(false);
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    switch (op) {
    case ast::BinaryExpression::Mul:
      return {
          /*lhs=*/mulOperandValue<ast::BinaryExpression>(true),
          /*rhs=*/mulOperandValue<ast::BinaryExpression>(true),
          /*output=*/copyInputToOutput<ast::BinaryExpression>(),
      };
    case ast::BinaryExpression::BitAnd:
    case ast::BinaryExpression::BitOr:
    case ast::BinaryExpression::BitXor:
      // A bitwise operator hands the 0/1 range of its operands on, so an
      // operand of one below a multiplication is still one of the
      // multiplication's.
      return {
          /*lhs=*/copyFromInput<ast::BinaryExpression>(),
          /*rhs=*/copyFromInput<ast::BinaryExpression>(),
          /*output=*/copyInputToOutput<ast::BinaryExpression>(),
      };
    default:
      return Super::getBinaryExpressionTransferFns(op);
    }
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return {
        /*target type=*/mulOperandValue<ast::CastExpression>(false),
        // A cast only widens the value, so a comparison below it is one the
        // multiplication still sees.
        /*operand=*/copyFromInput<ast::CastExpression>(),
        /*output=*/copyInputToOutput<ast::CastExpression>(),
    };
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return {
        // The condition is not the expression's value, the two values are.
        /*condition=*/mulOperandValue<ast::ConditionalExpression>(false),
        /*true value=*/copyFromInput<ast::ConditionalExpression>(),
        /*false value=*/copyFromInput<ast::ConditionalExpression>(),
        /*output=*/copyInputToOutput<ast::ConditionalExpression>(),
    };
  }

private:
  /// Returns a transfer function stating that a subelement's value is, or is
  /// not, the value of a multiplication's operand.
  template <typename ASTNode>
  static TransferFn<ASTNode, INPUT_DEPENDENCY>
  mulOperandValue(bool isMulOperandValue) {
    return TransferFn<ASTNode, INPUT_DEPENDENCY>(
        [isMulOperandValue](BooleanMulOperandTypingContext context) {
          context.isMulOperandValue = isMulOperandValue;
          return context;
        });
  }
};

} // namespace detail

/// Type system that keeps the generator from emitting operations whose
/// latency the compiler trivially optimizes away again.
///
/// Some fuzzing targets cares about the latency of the operations it generated
/// and for simplicity assume a 1:1 correspondence of C operation and handshake
/// operation.
/// Disabling the optimizations here removes the worst offenders of
/// optimizations that would break this assumption.
///
/// This type system is the conjunction of one sub type system per such
/// optimization, each of which knows exactly one pattern and constrains the
/// generator just enough for that pattern not to match.
class DisableLatencyOptTypeSystem final
    : public ConjunctionTypeSystemBase<DisableLatencyOptTypeSystem,
                                       detail::MulStrengthReductionTypeSystem,
                                       detail::MulSelectFoldTypeSystem> {
public:
  using ConjunctionTypeSystemBase::ConjunctionTypeSystemBase;

  ~DisableLatencyOptTypeSystem() override;
};

} // namespace dynamatic::gen

#endif
