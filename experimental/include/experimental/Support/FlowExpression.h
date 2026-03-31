#ifndef DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H
#define DYNAMATIC_SUPPORT_FLOW_EXPRESSION_H

#include "dynamatic/Analysis/IndexChannelAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/LinearAlgebra/Gaussian.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <variant>

// FlowExpression together with FlowVariable implements a DSL for combining flow
// variables, i.e. variables that track the flow and state of tokens, into
// equations. A similar DSL exists for constraint programming in
// `ConstraintProgramming.h`, but it is not reused for the following reasons:
// 1. FlowExpression uses integer coefficients, whereas CPVars have doubles as
// coefficients
// 2. Metadata that is only necessary for FlowExpressions can easily be added
// (variant, index tracker)
// 3. No name is necessary, as each variable is uniquely defined by the metadata
// 4. Dedicated conversion function to put multiple flow expressions into a
// matrix, with an ordering according to whether the variables are annotatable
// in SMV for use with Gaussian elimination

namespace dynamatic {
namespace handshake {

// IndexTracker is used to represent flow variables that, instead of tracking
// any token, only track tokens of a specific index. It is automatically added
// for channel lambdas using IndexChannelAnalysis to determine if the channel
// deals with indices.
struct IndexTracker {
  // numValues is the number of index values that are possible for this token.
  // For example, if the token is used as the select input of a mux with 3 data
  // inputs, numValues will be 3. The indices are always assumed to range from
  // (0 .. numValues - 1), so all possible tokens would be {0, 1, 2} in the
  // example.
  // Note that this is slightly more information than simply the bit width of a
  // channel: In the example, the bit width of the channel is 2 bits, but one of
  // the 4 resulting options is invalid.
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
        !mapper.map(SINGLE_VALUE_LIT, singleValue)) {
      llvm::report_fatal_error("json parsing of failed");
    }

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
  // been sent. It is stored as an shared pointer because it needs to be stored
  // as a pointer as InternalStateNamer is a virtual class, and because it could
  // be copied often.
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

  // Get the annotater for internal state
  // If the annotater does not exist, a nullptr is returned
  // Usage example:
  // if (auto annotater = var.getAnnotater()) {
  //   ...
  // }
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
    llvm::report_fatal_error("non-exhaustive pattern");
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

struct FlowEquationExtractor {
  std::vector<FlowExpression> equations;
  const IndexChannelAnalysis &indexChannelAnalysis;

  FlowEquationExtractor(const IndexChannelAnalysis &ica)
      : equations(), indexChannelAnalysis(ica) {}
  LogicalResult extractAll(ModuleOp modOp);

  LogicalResult extractSlotEquation(const FlowVariable &in,
                                    const FlowVariable &out,
                                    const FlowVariable &slot);
  LogicalResult extractEagerSentEquation(const FlowVariable &in,
                                         const FlowVariable &out,
                                         const FlowVariable &sent);

  LogicalResult extractArithmeticJoinOp(Operation &op);
  LogicalResult extractBranchOp(ConditionalBranchOp branchOp);
  LogicalResult extractBufferOp(BufferOp bufferOp);
  LogicalResult extractControlMergeOp(ControlMergeOp cmergeOp);
  LogicalResult extractEndOp(EndOp endOp);
  LogicalResult extractForkOp(ForkOp forkOp);
  LogicalResult extractLoadOp(LoadOp loadOp);
  LogicalResult extractMemoryControllerOp(MemoryControllerOp memCon);
  LogicalResult extractMuxOp(MuxOp muxOp);
  LogicalResult extractPipeline(LatencyInterface op, FlowVariable &internal);
  LogicalResult extractStoreOp(StoreOp storeOp);
};

namespace {
// This is a helper class to create a bidirectional mapping between variables
// and indices
class VariableRegistry {
public:
  size_t addVariable(const FlowVariable &var) {
    if (auto it = varToIndex.find(var); it != varToIndex.end()) {
      return it->second;
    }
    size_t newIdx = indexToVar.size();
    varToIndex[var] = newIdx;
    indexToVar.push_back(var);
    return newIdx;
  }

  bool verify() {
    if (!(varToIndex.size() == indexToVar.size()))
      return false;
    for (size_t i = 0; i < indexToVar.size(); ++i) {
      FlowVariable &a = indexToVar[i];
      size_t j = varToIndex[a];
      if (i != j)
        return false;
    }
    return true;
  }

  inline size_t getIndex(const FlowVariable &var) const {
    return varToIndex.at(var);
  }
  inline const FlowVariable &getVar(size_t idx) const {
    return indexToVar.at(idx);
  }
  inline size_t size() const { return indexToVar.size(); }

private:
  std::unordered_map<FlowVariable, size_t> varToIndex;
  std::vector<FlowVariable> indexToVar;
};
} // namespace

// FlowSystem is a wrapper for combining flow equations with a corresponding
// matrix. It keeps track of which variable corresponds to which column index,
// and makes sure that low indices are resesrved for variables that cannot be
// annotated to ensure they are eliminated first in the row-echelon form
struct FlowSystem {
  VariableRegistry registry;
  size_t nLambdas;
  MatIntType matrix;

  FlowExpression getRowAsExpression(size_t row) const;

  FlowSystem() = default;
  FlowSystem(const std::vector<FlowExpression> &exprs);
};
} // namespace handshake
} // namespace dynamatic

#endif
