#include "experimental/Support/FlowExpression.h"

namespace dynamatic {
namespace handshake {
FlowVariable::FlowVariable(const IndexChannelAnalysis &indexChannels,
                           ChannelLambda channel) {
  *this = FlowVariable(Variants(channel));
  if (auto numValues = indexChannels.getIndexChannelValues(channel.channel)) {
    indexTokenConstraint = *numValues;
  }
}

FlowVariable FlowVariable::nextInternal() const {
  auto *lambda = std::get_if<InternalLambda>(&variable);
  assert(lambda && "next internal can only be used on internal variables");
  InternalLambda nextLambda = *lambda;
  nextLambda.index += 1;
  FlowVariable next = *this;
  next.variable = nextLambda;
  return next;
}
FlowVariable FlowVariable::setTrackedTokens(size_t x) const {
  FlowVariable p = *this;
  if (!p.isIndex()) {
    p.getDebugName();
    assert(p.isIndex());
  }
  p.indexTokenConstraint->trackedValue = x;
  return p;
}

std::shared_ptr<InternalStateNamer> FlowVariable::getAnnotater() const {
  auto *namer = std::get_if<std::shared_ptr<InternalStateNamer>>(&variable);
  if (!namer)
    return nullptr;

  auto &state = *namer;
  if (indexTokenConstraint && indexTokenConstraint->trackedValue) {
    return state->tryConstrain(*(indexTokenConstraint->trackedValue));
  }
  return state;
}

std::string FlowVariable::getDebugName() const {
  std::string ret = "";
  if (auto *namer =
          std::get_if<std::shared_ptr<InternalStateNamer>>(&variable)) {
    ret += (*namer)->getSMVName();
  }
  if (auto *channel = std::get_if<ChannelLambda>(&variable)) {
    if (auto *op = channel->channel.getDefiningOp()) {
      ret += getUniqueName(op);
      for (auto [i, ch] : llvm::enumerate(op->getResults())) {
        if (ch == channel->channel) {
          ret += llvm::formatv(".out{0}", i);
          break;
        }
      }
    } else {
      for (auto &opop : channel->channel.getUses()) {
        ret += llvm::formatv("{0}.in{1}", getUniqueName(opop.getOwner()),
                             opop.getOperandNumber());
        break;
      }
    }
  }
  if (auto *internal = std::get_if<InternalLambda>(&variable)) {
    ret +=
        llvm::formatv("{0}.#{1}", getUniqueName(internal->op), internal->index);
  }

  if (indexTokenConstraint) {
    if (indexTokenConstraint->trackedValue) {
      ret += llvm::formatv("(={0})", *(indexTokenConstraint->trackedValue));
    } else {
      ret += llvm::formatv("(=x)");
    }
  }
  return ret;
}

FlowExpression::FlowExpression(const FlowVariable &v) {
  if (v.isIndex()) {
    if (v.indexTokenConstraint->trackedValue) {
      terms[v] = 1;
      return;
    }
    // If plusAndMinus, separate into plus and minus parts
    for (size_t i = 0; i < v.indexTokenConstraint->numValues; ++i) {
      terms[v.setTrackedTokens(i)] = 1;
    }
  } else {
    terms[v] = 1;
  }
};

llvm::json::Value FlowExpression::toJSON() const {
  std::vector<llvm::json::Value> jsonTerms{};
  for (auto &[key, value] : terms) {
    auto *namer =
        std::get_if<std::shared_ptr<InternalStateNamer>>(&key.variable);
    assert(namer);
    std::optional<llvm::json::Value> constraintJson;
    if (key.indexTokenConstraint) {
      constraintJson = key.indexTokenConstraint->toJSON();
    } else {
      constraintJson = nullptr;
    }
    // int pm = key.pm;
    jsonTerms.emplace_back(
        llvm::json::Object({{STATE_LIT, (*namer)->toJSON()},
                            {COEFFICIENT_LIT, value},
                            {CONSTRAINT_LIT, constraintJson}}));
  }
  return llvm::json::Array(jsonTerms);
}

FlowExpression FlowExpression::fromJSON(const llvm::json::Value &value,
                                        llvm::json::Path path) {
  FlowExpression expr;
  const llvm::json::Array *arr = value.getAsArray();
  assert(arr && "FlowExpression JSON is not an array");
  for (const llvm::json::Value &termJSON : *arr) {
    const llvm::json::Object *obj = termJSON.getAsObject();
    assert(obj && "FlowExpression term JSON not an object");
    const llvm::json::Value *state = obj->get(STATE_LIT);
    assert(state && "FlowExpression term JSON does not contain STATE_LIT");
    std::shared_ptr<InternalStateNamer> namer =
        InternalStateNamer::fromJSON(*state, path);
    FlowVariable var(namer);
    int coef;
    llvm::json::ObjectMapper mapper(termJSON, path);
    if (!mapper || !mapper.map(COEFFICIENT_LIT, coef)) {
      assert(false &&
             "FlowExpression term JSON does not contain COEFFICIENT_LIT");
    }
    const llvm::json::Value *constraint = obj->get(CONSTRAINT_LIT);
    assert(constraint && "FlowExpression does not contain CONSTRAINT_LIT");
    if (auto n = constraint->getAsNull()) {
      assert(!var.indexTokenConstraint);
    } else {
      var.indexTokenConstraint = IndexTracker::fromJSON(*constraint, path);
    }

    expr.terms[var] = coef;
  }
  return expr;
}

std::string FlowExpression::debug() const {
  std::string ret;
  for (auto &[var, coef] : terms) {
    if (coef == 0) {
      ret += "0 * ";
    } else if (coef == 1) {
      ret += "+ ";
    } else if (coef == -1) {
      ret += "- ";
    } else {
      ret += llvm::formatv("{0} * ", coef);
    }
    ret += var.getDebugName();
    ret += "  ";
  }
  ret += "\n";
  return ret;
}

FlowExpression operator-(FlowExpression expr) {
  for (auto &[key, value] : expr.terms) {
    expr.terms[key] = -value;
  }
  return expr;
}

FlowExpression operator+(FlowExpression left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] += value;
  }
  return left;
}

FlowExpression operator*(int coef, FlowExpression expr) {
  for (auto &[key, value] : expr.terms) {
    expr.terms[key] *= coef;
  }
  return expr;
}

FlowExpression operator-(FlowExpression left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] -= value;
  }
  return left;
}

void operator+=(FlowExpression &left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] += value;
  }
}

void operator-=(FlowExpression &left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] -= value;
  }
}

} // namespace handshake
} // namespace dynamatic
