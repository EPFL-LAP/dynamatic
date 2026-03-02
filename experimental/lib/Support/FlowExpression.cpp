#include "experimental/Support/FlowExpression.h"

namespace dynamatic {
namespace handshake {
FlowVariable::FlowVariable(const OpResult &channel) {
  op = channel.getDefiningOp();
  assert(op && "cannot get FlowVariable from channel without owner");
  type = outputLambda;
  lambdaIndex = channel.getResultNumber();

  pm = PLUSMINUS::notApplicable;
  if (auto ct = dyn_cast<handshake::ChannelType>(channel.getType())) {
    if (ct.getDataBitWidth() == 1) {
      pm = PLUSMINUS::plusAndMinus;
    }
  }
}

FlowVariable::FlowVariable(OpOperand &back, Operation &resOp) {
  Value channel = back.get();
  op = channel.getDefiningOp();
  pm = PLUSMINUS::notApplicable;
  if (auto ct = dyn_cast<handshake::ChannelType>(channel.getType())) {
    if (ct.getDataBitWidth() == 1) {
      pm = PLUSMINUS::plusAndMinus;
    }
  }
  if (op == nullptr) {
    type = inputLambda;
    op = &resOp;
    lambdaIndex = back.getOperandNumber();
  } else {
    type = outputLambda;
    bool found = false;
    for (auto res : op->getResults()) {
      if (res == channel) {
        assert(!found && "found multiple matches");
        found = true;
        lambdaIndex = res.getResultNumber();
      }
    }
    assert(found && "did not find matching OpResult");
  }
}

// utility functions for initializing variables
FlowVariable FlowVariable::internalChannel(Operation *op, unsigned index) {
  return FlowVariable(TYPE::internalLambda, op, index);
}

FlowVariable FlowVariable::nextInternal() const {
  assert(type == TYPE::internalLambda);
  FlowVariable next = *this;
  next.lambdaIndex = lambdaIndex + 1;
  return next;
}

bool FlowVariable::operator==(const FlowVariable &other) const {
  if (type == TYPE::internalState && other.type == TYPE::internalState) {
    return pm == other.pm && state.get() == other.state.get();
  }

  return type == other.type && lambdaIndex == other.lambdaIndex &&
         op == other.op && pm == other.pm;
}

bool FlowVariable::sameChannel(const FlowVariable &other) const {
  assert(isLambda());
  assert(other.isLambda());
  return type == other.type && lambdaIndex == other.lambdaIndex &&
         op == other.op;
}
std::shared_ptr<InternalStateNamer> FlowVariable::getAnnotater() const {
  if (isLambda()) {
    return nullptr;
  }
  assert(type == internalState);

  switch (pm) {
  case notApplicable:
    return state;
  case plus:
    return state->tryConstrain(1);
  case minus:
    return state->tryConstrain(0);
  case plusAndMinus:
    // should not happen because `plusAndMinus` is split into `plus` and `minus`
    // within a flow expression
    assert(false && "trying to get the annotater for plusAndMinus");
    return nullptr;
  }
}

std::string FlowVariable::getName() const {
  std::string sign;
  switch (pm) {
  case notApplicable:
    sign = "";
    break;
  case plus:
    sign = "+";
    break;
  case minus:
    sign = "-";
    break;
  case plusAndMinus:
    sign = "+-";
    break;
  }
  switch (type) {
  case internalState:
    return llvm::formatv("{0}{1}", state->getSMVName(), sign);
  case inputLambda:
    return llvm::formatv("{0}.in_{1}{2}", getUniqueName(op), lambdaIndex, sign)
        .str();
  case outputLambda:
    return llvm::formatv("{0}.out_{1}{2}", getUniqueName(op), lambdaIndex, sign)
        .str();
  case internalLambda:
    return llvm::formatv("{0}.#{1}{2}", getUniqueName(op), lambdaIndex, sign)
        .str();
  };
}

FlowExpression::FlowExpression(const FlowVariable &v) {
  if (v.isPlusMinus()) {
    // If plusAndMinus, separate into plus and minus parts
    terms[v.getPlus()] = 1;
    terms[v.getMinus()] = 1;
  } else {
    terms[v] = 1;
  }
};

void FlowExpression::debug() const {
  std::string txt = "0 = ";
  bool first = true;
  for (auto &[key, value] : terms) {
    if (!first) {
      if (value > 0) {
        txt += " + ";
      } else if (value < 0) {
        txt += " - ";
      }
    } else {
      if (value < 0)
        txt += "-";
      first = false;
    }
    if (abs(value) == 1) {
      txt += key.getName();

    } else {
      txt += llvm::formatv("{0} * {1}", value, key.getName()).str();
    }
  }
  llvm::errs() << txt << "\n";
}

llvm::json::Value FlowExpression::toJSON() const {
  std::vector<llvm::json::Value> jsonTerms{};
  for (auto &[key, value] : terms) {
    assert(key.type == FlowVariable::TYPE::internalState);
    int pm = key.pm;
    jsonTerms.emplace_back(llvm::json::Object({{STATE_LIT, key.state->toJSON()},
                                               {COEFFICIENT_LIT, value},
                                               {CONSTRAINT_LIT, pm}}));
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
    auto namer = InternalStateNamer::fromJSON(*state, path);
    FlowVariable var(std::move(namer));
    int coef;
    llvm::json::ObjectMapper mapper(termJSON, path);
    if (!mapper || !mapper.map(COEFFICIENT_LIT, coef)) {
      assert(false &&
             "FlowExpression term JSON does not contain COEFFICIENT_LIT");
    }
    int pm;
    if (!mapper.map(CONSTRAINT_LIT, pm)) {
      assert(false && "FlowExpression does not contain CONSTRAINT_LIT");
    }
    var.pm = (FlowVariable::PLUSMINUS)pm;

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
