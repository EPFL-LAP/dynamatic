//===- PropertyExpr.cpp - Expressions of formal properties ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the construction, JSON serialization, and SMV printing of the
// expressions of formal properties.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/PropertyExpr.h"
#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {
/// Builds a visitor of a `std::variant` out of one lambda per alternative.
template <typename... Fns>
struct Overloaded : Fns... {
  using Fns::operator()...;
};
template <typename... Fns>
Overloaded(Fns...) -> Overloaded<Fns...>;

/// Spelling of an operator, both in SMV and in JSON.
template <typename OpT>
struct OpSpelling {
  OpT op;
  StringLiteral json;
  StringLiteral smv;
};

constexpr OpSpelling<UnaryExpr::Op> UNARY_SPELLINGS[] = {
    {UnaryExpr::Not, "not", "!"}, {UnaryExpr::ToInt, "toint", "toint"},
    {UnaryExpr::AG, "AG", "AG"},  {UnaryExpr::AF, "AF", "AF"},
    {UnaryExpr::AX, "AX", "AX"},  {UnaryExpr::EG, "EG", "EG"},
    {UnaryExpr::EF, "EF", "EF"},  {UnaryExpr::EX, "EX", "EX"},
};

constexpr OpSpelling<BinaryExpr::Op> BINARY_SPELLINGS[] = {
    {BinaryExpr::Implies, "implies", "->"},
    {BinaryExpr::Iff, "iff", "<->"},
    {BinaryExpr::Sub, "sub", "-"},
    {BinaryExpr::Mul, "mul", "*"},
    {BinaryExpr::Eq, "eq", "="},
    {BinaryExpr::Lt, "lt", "<"},
    {BinaryExpr::Le, "le", "<="},
    {BinaryExpr::Gt, "gt", ">"},
    {BinaryExpr::Ge, "ge", ">="},
};

constexpr OpSpelling<NaryExpr::Op> NARY_SPELLINGS[] = {
    {NaryExpr::And, "and", "&"},
    {NaryExpr::Or, "or", "|"},
    {NaryExpr::Add, "add", "+"},
    {NaryExpr::Count, "count", "count"},
};

// Kinds of the leaves in JSON
constexpr StringLiteral STATE_KIND = "state";
constexpr StringLiteral SIGNAL_KIND = "signal";
constexpr StringLiteral INT_KIND = "int";
constexpr StringLiteral BOOL_KIND = "bool";

constexpr StringLiteral OP_LIT = "op";
constexpr StringLiteral OPERANDS_LIT = "operands";
constexpr StringLiteral STATE_LIT = "state";
constexpr StringLiteral CONSTRAINT_LIT = "constraint";
constexpr StringLiteral SIGNAL_LIT = "signal";
constexpr StringLiteral VALID_LIT = "valid";
constexpr StringLiteral READY_LIT = "ready";
constexpr StringLiteral UNIT_LIT = "unit";
constexpr StringLiteral SIDE_LIT = "side";
constexpr StringLiteral OPERAND_LIT = "operand";
constexpr StringLiteral RESULT_LIT = "result";
constexpr StringLiteral INDEX_LIT = "index";
constexpr StringLiteral PORT_LIT = "port";
constexpr StringLiteral VALUE_LIT = "value";
} // namespace

template <typename OpT, size_t N>
static const OpSpelling<OpT> &getSpelling(const OpSpelling<OpT> (&spellings)[N],
                                          OpT op) {
  for (const OpSpelling<OpT> &spelling : spellings)
    if (spelling.op == op)
      return spelling;
  llvm_unreachable("operator without spelling");
}

/// The operator spelled `json` in JSON, if any.
template <typename OpT, size_t N>
static std::optional<OpT> parseOp(const OpSpelling<OpT> (&spellings)[N],
                                  StringRef json) {
  for (const OpSpelling<OpT> &spelling : spellings)
    if (spelling.json == json)
      return spelling.op;
  return std::nullopt;
}

PropertyExpr dynamatic::conjunction(std::vector<PropertyExpr> operands) {
  if (operands.empty())
    return Constant::boolean(true);
  if (operands.size() == 1)
    return operands.front();
  return NaryExpr(NaryExpr::And, std::move(operands));
}

PropertyExpr dynamatic::disjunction(std::vector<PropertyExpr> operands) {
  if (operands.empty())
    return Constant::boolean(false);
  if (operands.size() == 1)
    return operands.front();
  return NaryExpr(NaryExpr::Or, std::move(operands));
}

PropertyExpr dynamatic::sum(std::vector<PropertyExpr> operands) {
  if (operands.empty())
    return Constant::integer(0);
  if (operands.size() == 1)
    return operands.front();
  return NaryExpr(NaryExpr::Add, std::move(operands));
}

void PropertyExpr::walk(
    llvm::function_ref<void(const PropertyExpr &)> fn) const {
  fn(*this);
  std::visit(Overloaded{
                 [&](const UnaryExpr &e) { e.getOperand().walk(fn); },
                 [&](const BinaryExpr &e) {
                   e.getLhs().walk(fn);
                   e.getRhs().walk(fn);
                 },
                 [&](const NaryExpr &e) {
                   for (const PropertyExpr &operand : e.getOperands())
                     operand.walk(fn);
                 },
                 [](const auto &leaf) {},
             },
             getNode());
}

bool PropertyExpr::isTemporal() const {
  return std::visit(
      Overloaded{
          [](const UnaryExpr &e) {
            return UnaryExpr::isTemporal(e.getOp()) ||
                   e.getOperand().isTemporal();
          },
          [](const BinaryExpr &e) {
            return e.getLhs().isTemporal() || e.getRhs().isTemporal();
          },
          [](const NaryExpr &e) {
            return llvm::any_of(e.getOperands(), [](const PropertyExpr &o) {
              return o.isTemporal();
            });
          },
          [](const auto &leaf) { return false; },
      },
      getNode());
}

std::string PropertyExpr::toSMV() const {
  return std::visit(
      Overloaded{
          [](const State &state) -> std::string {
            if (!state.getConstraint())
              return state.getNamer()->getSMVName();
            std::unique_ptr<ConstrainedNamer> constrained =
                state.getNamer()->tryConstrain(*state.getConstraint());
            assert(constrained &&
                   "state cannot be constrained to a token value");
            // The name of a constrained state is a conjunction (e.g.,
            // `fork0.outs_0_sent & (fork0.ins = TRUE)`), so it is
            // parenthesized like any other operation
            return "(" + constrained->getSMVName() + ")";
          },
          [](const Signal &signal) -> std::string {
            return llvm::formatv(
                "{0}.{1}_{2}", signal.getOpName(), signal.getPortName(),
                signal.getKind() == Signal::Valid ? "valid" : "ready");
          },
          [](const Constant &constant) -> std::string {
            if (const bool *b = std::get_if<bool>(&constant.getValue()))
              return *b ? "TRUE" : "FALSE";
            return std::to_string(std::get<int64_t>(constant.getValue()));
          },
          [](const UnaryExpr &e) -> std::string {
            StringRef op = getSpelling(UNARY_SPELLINGS, e.getOp()).smv;
            // toint is function-like, so it already delimits its operand
            if (e.getOp() == UnaryExpr::ToInt)
              return llvm::formatv("{0}({1})", op, e.getOperand().toSMV());
            return llvm::formatv("({0} {1})", op, e.getOperand().toSMV());
          },
          [](const BinaryExpr &e) -> std::string {
            return llvm::formatv("({0} {1} {2})", e.getLhs().toSMV(),
                                 getSpelling(BINARY_SPELLINGS, e.getOp()).smv,
                                 e.getRhs().toSMV());
          },
          [](const NaryExpr &e) -> std::string {
            StringRef op = getSpelling(NARY_SPELLINGS, e.getOp()).smv;
            std::vector<std::string> operands;
            for (const PropertyExpr &operand : e.getOperands())
              operands.push_back(operand.toSMV());
            // count is function-like, so it already delimits its operands
            if (e.getOp() == NaryExpr::Count)
              return llvm::formatv("{0}({1})", op, llvm::join(operands, ", "));
            return llvm::formatv("({0})",
                                 llvm::join(operands, (" " + op + " ").str()));
          },
      },
      getNode());
}

llvm::json::Value dynamatic::toJSON(const PropertyExpr &expr) {
  return std::visit(
      Overloaded{
          [](const State &state) -> llvm::json::Value {
            llvm::json::Object obj{{OP_LIT, STATE_KIND},
                                   {STATE_LIT, toJSON(state.getNamer())}};
            if (state.getConstraint())
              obj[CONSTRAINT_LIT] = *state.getConstraint();
            return obj;
          },
          [](const Signal &signal) -> llvm::json::Value {
            return llvm::json::Object{
                {OP_LIT, SIGNAL_KIND},
                {SIGNAL_LIT,
                 signal.getKind() == Signal::Valid ? VALID_LIT : READY_LIT},
                {UNIT_LIT, signal.getOpName()},
                {SIDE_LIT, signal.getSide() == Signal::Operand ? OPERAND_LIT
                                                               : RESULT_LIT},
                {INDEX_LIT, signal.getPortIndex()},
                {PORT_LIT, signal.getPortName()}};
          },
          [](const Constant &constant) -> llvm::json::Value {
            if (const bool *b = std::get_if<bool>(&constant.getValue()))
              return llvm::json::Object{{OP_LIT, BOOL_KIND}, {VALUE_LIT, *b}};
            return llvm::json::Object{
                {OP_LIT, INT_KIND},
                {VALUE_LIT, std::get<int64_t>(constant.getValue())}};
          },
          [](const UnaryExpr &e) -> llvm::json::Value {
            return llvm::json::Object{
                {OP_LIT, getSpelling(UNARY_SPELLINGS, e.getOp()).json},
                {OPERANDS_LIT, llvm::json::Array{e.getOperand()}}};
          },
          [](const BinaryExpr &e) -> llvm::json::Value {
            return llvm::json::Object{
                {OP_LIT, getSpelling(BINARY_SPELLINGS, e.getOp()).json},
                {OPERANDS_LIT, llvm::json::Array{e.getLhs(), e.getRhs()}}};
          },
          [](const NaryExpr &e) -> llvm::json::Value {
            return llvm::json::Object{
                {OP_LIT, getSpelling(NARY_SPELLINGS, e.getOp()).json},
                {OPERANDS_LIT, llvm::json::Array(e.getOperands())}};
          },
      },
      expr.getNode());
}

bool dynamatic::fromJSON(const llvm::json::Value &value, PropertyExpr &expr,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  std::string op;
  if (!mapper || !mapper.map(OP_LIT, op))
    return false;

  // Leaves
  if (op == STATE_KIND) {
    std::shared_ptr<InternalStateNamer> namer;
    std::optional<int32_t> constraint;
    if (!mapper.map(STATE_LIT, namer) ||
        !mapper.mapOptional(CONSTRAINT_LIT, constraint))
      return false;
    expr = State(std::move(namer), constraint);
    return true;
  }
  if (op == SIGNAL_KIND) {
    std::string kind, opName, side, portName;
    unsigned portIndex;
    if (!mapper.map(SIGNAL_LIT, kind) || !mapper.map(UNIT_LIT, opName) ||
        !mapper.map(SIDE_LIT, side) || !mapper.map(INDEX_LIT, portIndex) ||
        !mapper.map(PORT_LIT, portName))
      return false;
    if (kind != VALID_LIT && kind != READY_LIT) {
      path.field(SIGNAL_LIT).report("expected valid or ready");
      return false;
    }
    if (side != OPERAND_LIT && side != RESULT_LIT) {
      path.field(SIDE_LIT).report("expected operand or result");
      return false;
    }
    expr = Signal(kind == VALID_LIT ? Signal::Valid : Signal::Ready, opName,
                  side == OPERAND_LIT ? Signal::Operand : Signal::Result,
                  portIndex, portName);
    return true;
  }
  if (op == INT_KIND) {
    int64_t intValue;
    if (!mapper.map(VALUE_LIT, intValue))
      return false;
    expr = Constant::integer(intValue);
    return true;
  }
  if (op == BOOL_KIND) {
    bool boolValue;
    if (!mapper.map(VALUE_LIT, boolValue))
      return false;
    expr = Constant::boolean(boolValue);
    return true;
  }

  // Operations
  std::vector<PropertyExpr> operands;
  if (!mapper.map(OPERANDS_LIT, operands))
    return false;
  auto checkNumOperands = [&](size_t expected) {
    if (operands.size() == expected)
      return true;
    path.field(OPERANDS_LIT).report("wrong number of operands");
    return false;
  };
  if (std::optional<UnaryExpr::Op> unaryOp = parseOp(UNARY_SPELLINGS, op)) {
    if (!checkNumOperands(1))
      return false;
    expr = UnaryExpr(*unaryOp, operands[0]);
    return true;
  }
  if (std::optional<BinaryExpr::Op> binaryOp = parseOp(BINARY_SPELLINGS, op)) {
    if (!checkNumOperands(2))
      return false;
    expr = BinaryExpr(operands[0], *binaryOp, operands[1]);
    return true;
  }
  if (std::optional<NaryExpr::Op> naryOp = parseOp(NARY_SPELLINGS, op)) {
    if (operands.empty()) {
      path.field(OPERANDS_LIT).report("operation without operands");
      return false;
    }
    expr = NaryExpr(*naryOp, std::move(operands));
    return true;
  }
  path.field(OP_LIT).report("unknown operator");
  return false;
}
