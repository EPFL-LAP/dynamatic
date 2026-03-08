#include "experimental/Support/FlowExpression.h"

namespace dynamatic {
namespace handshake {
FlowVariable::FlowVariable(ChannelLambda channel,
                           const llvm::DenseMap<mlir::Value, IndexInfo> &map) {
  *this = FlowVariable(Variants(channel));
  for (auto &[key, value] : map) {
    if (key == channel.channel) {
      constraint = IndexConstraint(value);
    }
  }
}

FlowVariable FlowVariable::nextInternal() const {
  auto *lambda = std::get_if<InternalLambda>(&variable);
  assert(lambda && "next internal can only be used on internal variables");
  InternalLambda next = *lambda;
  next.index += 1;
  return next;
}

std::shared_ptr<InternalStateNamer> FlowVariable::getAnnotater() const {
  auto *namer = std::get_if<std::shared_ptr<InternalStateNamer>>(&variable);
  if (!namer)
    return nullptr;

  auto &state = *namer;
  if (constraint && constraint->singleValue) {
    return state->tryConstrain(*(constraint->singleValue));
  }
  return state;
}

FlowExpression::FlowExpression(const FlowVariable &v) {
  if (v.isIndex()) {
    if (v.constraint->singleValue) {
      terms[v] = 1;
      return;
    }
    // If plusAndMinus, separate into plus and minus parts
    for (size_t i = 0; i < v.constraint->info.numValues; ++i) {
      terms[v.getConstrained(i)] = 1;
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
    if (key.constraint) {
      constraintJson = key.constraint->toJSON();
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
      assert(!var.constraint);
    } else {
      var.constraint = IndexConstraint::fromJSON(*constraint, path);
    }

    expr.terms[var] = coef;
  }
  return expr;
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
