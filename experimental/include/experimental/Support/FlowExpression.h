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

struct IndexInfo {
  size_t numValues;
  IndexInfo(size_t numValues) : numValues(numValues) {}
  inline bool operator==(const IndexInfo &other) const {
    return numValues == other.numValues;
  }
};

struct IndexConstraint {
  IndexInfo info;
  // if singleValue is nullptr, the index is not constrained
  std::optional<size_t> singleValue;

  IndexConstraint(IndexInfo info) : info(info) {}
  inline bool operator==(const IndexConstraint &other) const {
    return info == other.info && singleValue == other.singleValue;
  }

  inline llvm::json::Value toJSON() const {
    return llvm::json::Object(
        {{NUM_VALUES_LIT, info.numValues}, {SINGLE_VALUE_LIT, singleValue}});
  }

  inline IndexConstraint static fromJSON(const llvm::json::Value &value,
                                         llvm::json::Path path) {
    size_t numValues;
    std::optional<size_t> singleValue;
    llvm::json::ObjectMapper mapper(value, path);
    if (!mapper || !mapper.map(NUM_VALUES_LIT, numValues) ||
        !mapper.map(SINGLE_VALUE_LIT, singleValue))
      assert(false && "json parsing failed");

    IndexConstraint ret(numValues);
    ret.singleValue = singleValue;
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

using Variants = std::variant<std::shared_ptr<InternalStateNamer>,
                              ChannelLambda, InternalLambda>;

struct FlowVariable {
  // A Lambda variable is defined by type, lambdaIndex, and op.
  // An internal state is defined by type, state
  Variants variable;
  std::optional<IndexConstraint> constraint;

  FlowVariable(Variants variable)
      : variable(std::move(variable)), constraint() {}
  FlowVariable(ChannelLambda l, const DenseMap<mlir::Value, IndexInfo> &map);
  FlowVariable(InternalLambda l) : FlowVariable(Variants(l)) {}
  FlowVariable(std::shared_ptr<InternalStateNamer> n)
      : FlowVariable(Variants(n)) {}

  // useful for generating multiple internal channels without collisions
  FlowVariable nextInternal() const;

  // compares the relevant struct fields to determine if two variables are equal
  inline bool operator==(const FlowVariable &other) const {
    return variable == other.variable && constraint == other.constraint;
  }

  // utility functions for handling binary channels
  inline bool isIndex() const { return constraint.has_value(); }
  inline FlowVariable getConstrained(size_t x) const {
    FlowVariable p = *this;
    if (!p.isIndex()) {
      p.debug();
      assert(p.isIndex());
    }
    p.constraint->singleValue = x;
    return p;
  }

  inline bool isLambda() const {
    return std::get_if<ChannelLambda>(&variable) ||
           std::get_if<InternalLambda>(&variable);
  }

  inline void debug() const {
    if (auto *namer =
            std::get_if<std::shared_ptr<InternalStateNamer>>(&variable)) {
      llvm::errs() << (*namer)->getSMVName();
    }
    if (auto *channel = std::get_if<ChannelLambda>(&variable)) {
      if (auto *op = channel->channel.getDefiningOp()) {
        llvm::errs() << getUniqueName(op);
        for (auto [i, ch] : llvm::enumerate(op->getResults())) {
          if (ch == channel->channel) {
            llvm::errs() << llvm::formatv(".out{0}", i);
            break;
          }
        }
      } else {
        for (auto &opop : channel->channel.getUses()) {
          llvm::errs() << llvm::formatv("{0}.in{1}",
                                        getUniqueName(opop.getOwner()),
                                        opop.getOperandNumber());
          break;
        }
      }
    }
    if (auto *internal = std::get_if<InternalLambda>(&variable)) {
      llvm::errs() << llvm::formatv("{0}.#{1}", getUniqueName(internal->op),
                                    internal->index);
    }

    if (constraint) {
      if (constraint->singleValue) {
        llvm::errs() << llvm::formatv("(={0})", *(constraint->singleValue));
      } else {
        llvm::errs() << llvm::formatv("(=x)");
      }
    }
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
    if (var.constraint) {
      return hash<Variants>()(var.variable) ^ hash<unsigned>()(0) ^
             hash<size_t>()(var.constraint->info.numValues) ^
             hash<std::optional<size_t>>()(var.constraint->singleValue);
    }
    return hash<Variants>()(var.variable) ^ hash<unsigned>()(1);
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

  inline void debug() {
    for (auto &[var, coef] : terms) {
      if (coef == 0) {
        llvm::errs() << "0 * ";
      } else if (coef == 1) {
        llvm::errs() << "+ ";
      } else if (coef == -1) {
        llvm::errs() << "- ";
      } else {
        llvm::errs() << llvm::formatv("{0} * ", coef);
      }
      var.debug();
      llvm::errs() << "  ";
    }
    llvm::errs() << "\n";
  }
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
