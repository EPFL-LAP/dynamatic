#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

namespace dynamatic {
namespace handshake {

struct OtherVariable {
  virtual ~OtherVariable() = default;
  virtual bool isAnnotatable() = 0;
  virtual std::string annotate() = 0;
  virtual llvm::json::Value toJSON() = 0;
  virtual std::optional<int64_t> getConstraint() = 0;
  virtual std::unique_ptr<OtherVariable> constrain(int64_t value) = 0;
  virtual std::unique_ptr<OtherVariable> getUnconstrained() = 0;
};

struct FlowVariable {
  enum TYPE { internalState, inputLambda, outputLambda, internalLambda };
  enum PLUSMINUS { notApplicable, plusAndMinus, plus, minus };
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

  FlowVariable(const OpResult &channel);

  FlowVariable(OpOperand &back, Operation &resOp);
  // utility functions for initializing variables
  static FlowVariable internalChannel(Operation *op, unsigned index);

  FlowVariable nextInternal() const;

  bool operator==(const FlowVariable &other) const;

  bool sameChannel(const FlowVariable &other) const;

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

  std::shared_ptr<InternalStateNamer> getAnnotater() const;
  std::string getName() const;
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
  void debug() const;

  llvm::json::Value toJSON() const;
  static FlowExpression fromJSON(const llvm::json::Value &value,
                                 llvm::json::Path path);

  inline static const StringLiteral COEFFICIENT_LIT = "coefficient";
  inline static const StringLiteral STATE_LIT = "state";
};

FlowExpression operator-(FlowExpression expr);

FlowExpression operator+(FlowExpression left, const FlowExpression &right);

FlowExpression operator*(int coef, FlowExpression expr);

FlowExpression operator-(FlowExpression left, const FlowExpression &right);

void operator+=(FlowExpression &left, const FlowExpression &right);

void operator-=(FlowExpression &left, const FlowExpression &right);
} // namespace handshake
} // namespace dynamatic
