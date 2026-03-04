#ifndef DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H
#define DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <variant>

namespace dynamatic {
namespace handshake {

struct ChannelLambda {
  mlir::Value channel;

  ChannelLambda(mlir::Value channel) : channel(channel) {}
  inline bool operator==(const ChannelLambda &other) const {
    return channel == other.channel;
  }
};

struct InternalLambda {
  Operation *op;
  unsigned index;

  InternalLambda(Operation *op, unsigned index) : op(op), index(index) {}
  inline bool operator==(const InternalLambda &other) const {
    return op == other.op && index == other.index;
  }
};

using Variants = std::variant<std::shared_ptr<InternalStateNamer>,
                              ChannelLambda, InternalLambda>;

struct FlowVariable {
  enum PLUSMINUS { notApplicable = 0, plusAndMinus = -1, plus = 1, minus = 2 };
  // A Lambda variable is defined by type, lambdaIndex, and op.
  // An internal state is defined by type, state
  Variants variable;
  PLUSMINUS pm;

  FlowVariable(Variants variable)
      : variable(std::move(variable)), pm(PLUSMINUS::notApplicable) {}
  FlowVariable(ChannelLambda l);
  FlowVariable(InternalLambda l) : FlowVariable(Variants(l)) {}
  FlowVariable(std::shared_ptr<InternalStateNamer> n)
      : FlowVariable(Variants(n)) {}

  /*
  FlowVariable(std::shared_ptr<InternalStateNamer> state)
      : variable(TYPE::internalState), pm(PLUSMINUS::notApplicable),
        state(std::move(state)) {}
  FlowVariable(TYPE t, Operation *op, unsigned lambdaIndex)
      : type(t), lambdaIndex(lambdaIndex), op(op),
        pm(PLUSMINUS::notApplicable) {
    assert(
        t != TYPE::internalState &&
        "internal states should be initialized with the according constructor");
  }

  // FlowVariable from output
  FlowVariable(const OpResult &channel);

  // FlowVariable from input
  FlowVariable(OpOperand &back, Operation &resOp);

  // utility functions for initializing internal channels
  static FlowVariable internalChannel(Operation *op, unsigned index);
  */

  // useful for generating multiple internal channels without collisions
  FlowVariable nextInternal() const;

  // compares the relevant struct fields to determine if two variables are equal
  inline bool operator==(const FlowVariable &other) const {
    return variable == other.variable && pm == other.pm;
  }

  // utility functions for handling binary channels
  inline bool isPlusMinus() const { return pm == plusAndMinus; }
  inline FlowVariable getPlus() const {
    assert(isPlusMinus());
    FlowVariable p = *this;
    p.pm = plus;
    return p;
  }

  inline FlowVariable getMinus() const {
    assert(isPlusMinus());
    FlowVariable p = *this;
    p.pm = minus;
    return p;
  }

  inline bool isLambda() const {
    return std::get_if<ChannelLambda>(&variable) ||
           std::get_if<InternalLambda>(&variable);
  }

  // get the annotater for internal state - if it exists
  std::shared_ptr<InternalStateNamer> getAnnotater() const;
};

} // namespace handshake
} // namespace dynamatic

using namespace dynamatic;
using namespace dynamatic::handshake;
template <>
struct std::hash<Variants> {
  size_t operator()(const Variants &vars) const {
    using std::hash;
    size_t chunk = hash<size_t>()(vars.index());
    if (auto *namer = std::get_if<std::shared_ptr<InternalStateNamer>>(&vars)) {
      return chunk ^ hash<std::string>()((*namer)->getSMVName());
    }
    if (auto *channel = std::get_if<ChannelLambda>(&vars)) {
      return chunk ^ mlir::hash_value(channel->channel);
    }
    if (auto *internal = std::get_if<InternalLambda>(&vars)) {
      return chunk ^ hash<Operation *>()(internal->op) ^
             hash<unsigned>()(internal->index);
    }
    assert(false && "is pattern not exhaustive?");
  }
};
// Hash implementation required so that FlowVariable can be used in an
// unordered_map
template <>
struct std::hash<FlowVariable> {
  size_t operator()(const FlowVariable &var) const {
    using std::hash;
    return hash<Variants>()(var.variable) ^
           hash<FlowVariable::PLUSMINUS>()(var.pm);
  }
};

namespace dynamatic {
namespace handshake {
struct FlowExpression {
  std::unordered_map<FlowVariable, int> terms;
  FlowExpression() = default;
  FlowExpression(const FlowVariable &v);

  llvm::json::Value toJSON() const;
  static FlowExpression fromJSON(const llvm::json::Value &value,
                                 llvm::json::Path path);

  inline static const StringLiteral COEFFICIENT_LIT = "coefficient";
  inline static const StringLiteral STATE_LIT = "state";
  inline static const StringLiteral CONSTRAINT_LIT = "constraint";
};

FlowExpression operator-(FlowExpression expr);

FlowExpression operator+(FlowExpression left, const FlowExpression &right);

FlowExpression operator*(int coef, FlowExpression expr);

FlowExpression operator-(FlowExpression left, const FlowExpression &right);

void operator+=(FlowExpression &left, const FlowExpression &right);

void operator-=(FlowExpression &left, const FlowExpression &right);
} // namespace handshake
} // namespace dynamatic

#endif
