#ifndef DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H
#define DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H

#include "dynamatic/Analysis/IndexChannelAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <variant>

namespace dynamatic {
namespace handshake {

struct IndexTracker {
  size_t numValues;
  // Which token value does this variable track?
  // e.g. 1 to only count tokens with value 1
  // If trackedValue has no value, any value (0..numValues) is tracked
  std::optional<size_t> trackedValue;

  IndexTracker(size_t numValues) : numValues(numValues) {}
  inline bool operator==(const IndexTracker &other) const {
    return numValues == other.numValues && trackedValue == other.trackedValue;
  }

  inline llvm::json::Value toJSON() const {
    return llvm::json::Object(
        {{NUM_VALUES_LIT, numValues}, {SINGLE_VALUE_LIT, trackedValue}});
  }

  inline IndexTracker static fromJSON(const llvm::json::Value &value,
                                      llvm::json::Path path) {
    size_t numValues;
    std::optional<size_t> singleValue;
    llvm::json::ObjectMapper mapper(value, path);
    if (!mapper || !mapper.map(NUM_VALUES_LIT, numValues) ||
        !mapper.map(SINGLE_VALUE_LIT, singleValue))
      assert(false && "json parsing failed");

    IndexTracker ret(numValues);
    ret.trackedValue = singleValue;
    return ret;
  }

  inline static const StringLiteral NUM_VALUES_LIT = "num_values";
  inline static const StringLiteral SINGLE_VALUE_LIT = "single_value";
};

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

struct FlowVariable {
  // A flow variable can be defined by different parts of a circuit:
  // 1. It can be a channel lambda, which tracks the number of tokens that
  // propagated through a specific channel.
  // 2. It can be an internal state, which represents any data that can be found
  // in the final HDL: Most commonly, this is the data within a slot of a
  // buffer, or the state keeping track of which results of an eager fork have
  // been sent.
  // 3. It can be an internal lambda, which acts similar to a channel lambda,
  // except it does not count the tokens of a real channel in the handshake
  // MLIR, but rather a fictional channel within an operation (e.g. a channel
  // between slots of a buffer with multiple slots)
  using Variants = std::variant<std::shared_ptr<InternalStateNamer>,
                                ChannelLambda, InternalLambda>;
  Variants variable;

  // When indexTokenConstraint is set, only tokens of a specific value are
  // tracked. This can be useful e.g. in the case of control merges, where it is
  // known that a token at operand 0 leads to a token of value 0 at the index
  // output, and correspondingly for a token at operand 1.
  std::optional<IndexTracker> indexTokenConstraint;

  FlowVariable(Variants variable)
      : variable(std::move(variable)), indexTokenConstraint() {}
  FlowVariable(const IndexChannelAnalysis &indexChannels,
               ChannelLambda channel);
  FlowVariable(InternalLambda l) : FlowVariable(Variants(l)) {}
  FlowVariable(std::shared_ptr<InternalStateNamer> n)
      : FlowVariable(Variants(n)) {}

  // Utility function for generating multiple internal channels for a single
  // operation without collisions of indices
  FlowVariable nextInternal() const;

  // Compares the relevant struct fields to determine if two variables are equal
  inline bool operator==(const FlowVariable &other) const {
    return variable == other.variable &&
           indexTokenConstraint == other.indexTokenConstraint;
  }

  // Utility functions for handling flow variables with index tokens
  inline bool isIndex() const { return indexTokenConstraint.has_value(); }
  FlowVariable setTrackedTokens(size_t x) const;

  std::string getDebugName() const;

  // get the annotater for internal state - if it exists
  std::shared_ptr<InternalStateNamer> getAnnotater() const;
};

} // namespace handshake
} // namespace dynamatic

using namespace dynamatic;
using namespace dynamatic::handshake;
template <>
struct std::hash<FlowVariable::Variants> {
  size_t operator()(const FlowVariable::Variants &vars) const {
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
    if (var.indexTokenConstraint) {
      return hash<FlowVariable::Variants>()(var.variable) ^
             hash<unsigned>()(0) ^
             hash<size_t>()(var.indexTokenConstraint->numValues) ^
             hash<std::optional<size_t>>()(
                 var.indexTokenConstraint->trackedValue);
    }
    return hash<FlowVariable::Variants>()(var.variable) ^ hash<unsigned>()(1);
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

  std::string debug() const;

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
