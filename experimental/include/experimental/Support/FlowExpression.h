#ifndef DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H
#define DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

namespace dynamatic {
namespace handshake {

struct FlowVariable {
  enum TYPE { internalState, inputLambda, outputLambda, internalLambda };
  enum PLUSMINUS { notApplicable = 0, plusAndMinus = -1, plus = 1, minus = 2 };
  // A Lambda variable is defined by type, lambdaIndex, and op.
  // An internal state is defined by type, state
  TYPE type;
  unsigned lambdaIndex;
  Operation *op;
  PLUSMINUS pm;
  std::shared_ptr<InternalStateNamer> state;

  FlowVariable(std::shared_ptr<InternalStateNamer> state)
      : type(TYPE::internalState), pm(PLUSMINUS::notApplicable),
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

  // useful for generating multiple internal channels without collisions
  FlowVariable nextInternal() const;

  // compares the relevant struct fields to determine if two variables are equal
  bool operator==(const FlowVariable &other) const;

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
    return type == FlowVariable::TYPE::inputLambda ||
           type == FlowVariable::TYPE::outputLambda ||
           type == FlowVariable::TYPE::internalLambda;
  }

  // get the annotater for internal state - if it exists
  std::shared_ptr<InternalStateNamer> getAnnotater() const;
};

} // namespace handshake
} // namespace dynamatic

using namespace dynamatic;
using namespace dynamatic::handshake;
// Hash implementation required so that FlowVariable can be used in an
// unordered_map
template <>
struct std::hash<FlowVariable> {
  size_t operator()(const FlowVariable &var) const {
    using std::hash;
    if (var.type == FlowVariable::TYPE::internalState) {
      return hash<FlowVariable::TYPE>()(var.type) ^
             hash<std::string>()(var.state->getSMVName());
    }
    return (hash<FlowVariable::TYPE>()(var.type) ^
            hash<unsigned>()(var.lambdaIndex) ^ hash<Operation *>()(var.op));
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
