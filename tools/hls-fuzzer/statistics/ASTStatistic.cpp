#include "ASTStatistic.h"

#include "hls-fuzzer/ASTVisitor.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace dynamatic;

using ExpressionKey = gen::AbstractTypeSystem::ExpressionKey;
using StatementKey = gen::AbstractTypeSystem::StatementKey;
using NodeKey = std::variant<ExpressionKey, StatementKey>;

namespace {
/// Visitor counting how many expressions and statements of each kind occur in
/// an AST. The counts are accumulated into the map passed at construction,
/// keyed at the granularity of 'ExpressionKey' and 'StatementKey'.
class NodeCounter : public ASTVisitor<NodeCounter> {
public:
  using Super::visit;

  explicit NodeCounter(std::map<NodeKey, std::size_t> &counts)
      : counts(counts) {}

  void visit(const ast::Constant &node) {
    counts[ExpressionKey(ast::Constant::Tag{})]++;
    Super::visit(node);
  }

  void visit(const ast::Variable &node) {
    counts[ExpressionKey(ast::Variable::Tag{})]++;
    Super::visit(node);
  }

  void visit(const ast::BinaryExpression &node) {
    counts[ExpressionKey(node.getOp())]++;
    Super::visit(node);
  }

  void visit(const ast::CastExpression &node) {
    counts[ExpressionKey(ast::CastExpression::Tag{})]++;
    Super::visit(node);
  }

  void visit(const ast::UnaryExpression &node) {
    counts[ExpressionKey(node.getOp())]++;
    Super::visit(node);
  }

  void visit(const ast::ConditionalExpression &node) {
    counts[ExpressionKey(ast::ConditionalExpression::Tag{})]++;
    Super::visit(node);
  }

  void visit(const ast::ArrayReadExpression &node) {
    counts[ExpressionKey(ast::ArrayReadExpression::Tag{})]++;
    Super::visit(node);
  }

  void visit(const ast::ArrayAssignmentStatement &node) {
    counts[StatementKey(ast::ArrayAssignmentStatement::Tag{})]++;
    Super::visit(node);
  }

  void visit(const ast::StructuredForStatement &node) {
    counts[StatementKey(ast::StructuredForStatement::Tag{})]++;
    Super::visit(node);
  }

private:
  std::map<NodeKey, std::size_t> &counts;
};
} // namespace

static llvm::StringRef getName(ast::BinaryExpression::Op op) {
  using Op = ast::BinaryExpression::Op;
  switch (op) {
  case Op::BitAnd:
    return "&";
  case Op::BitOr:
    return "|";
  case Op::BitXor:
    return "^";
  case Op::ShiftLeft:
    return "<<";
  case Op::ShiftRight:
    return ">>";
  case Op::Plus:
    return "+";
  case Op::Minus:
    return "-";
  case Op::Mul:
    return "*";
  case Op::Greater:
    return ">";
  case Op::GreaterEqual:
    return ">=";
  case Op::Less:
    return "<";
  case Op::LessEqual:
    return "<=";
  case Op::Equal:
    return "==";
  case Op::NotEqual:
    return "!=";
  }
  llvm_unreachable("all binary operators handled");
}

static llvm::StringRef getName(ast::UnaryExpression::Op op) {
  using Op = ast::UnaryExpression::Op;
  switch (op) {
  case Op::BitwiseNot:
    return "~";
  case Op::BoolNot:
    return "!";
  case Op::Minus:
    return "-";
  }
  llvm_unreachable("all unary operators handled");
}

static std::string getName(const ExpressionKey &key) {
  return llvm::TypeSwitch<ExpressionKey, std::string>(key)
      .Case([](const ast::Constant::Tag *) { return "Constant"; })
      .Case([](const ast::Variable::Tag *) { return "Variable"; })
      .Case([](const ast::CastExpression::Tag *) { return "CastExpression"; })
      .Case([](const ast::ArrayReadExpression::Tag *) {
        return "ArrayReadExpression";
      })
      .Case([](const ast::ConditionalExpression::Tag *) {
        return "ConditionalExpression";
      })
      .Case([](const ast::BinaryExpression::Op *op) {
        return "BinaryExpression(" + getName(*op).str() + ")";
      })
      .Case([](const ast::UnaryExpression::Op *op) {
        return "UnaryExpression(" + getName(*op).str() + ")";
      });
}

static std::string getName(const StatementKey &key) {
  return llvm::TypeSwitch<StatementKey, std::string>(key)
      .Case([](const ast::ArrayAssignmentStatement::Tag *) {
        return "ArrayAssignmentStatement";
      })
      .Case([](const ast::StructuredForStatement::Tag *) {
        return "StructuredForStatement";
      });
}

/// Returns a human-readable name for 'key'.
static std::string getName(const NodeKey &key) {
  return llvm::TypeSwitch<NodeKey, std::string>(key)
      .Case([](const ExpressionKey *key) { return getName(*key); })
      .Case([](const StatementKey *key) { return getName(*key); });
}

/// Returns the node kind that 'getName' names 'name', or nullptr if no tracked
/// node kind has that name. The node kinds listed here must be kept in sync
/// with the ones counted by 'NodeCounter'.
static const NodeKey *lookupKey(llvm::StringRef name) {
  static const llvm::StringMap<NodeKey> keysByName = [] {
    llvm::StringMap<NodeKey> result;
    auto add = [&](NodeKey key) { result.try_emplace(getName(key), key); };

    add(ExpressionKey(ast::Constant::Tag{}));
    add(ExpressionKey(ast::Variable::Tag{}));
    add(ExpressionKey(ast::CastExpression::Tag{}));
    add(ExpressionKey(ast::ConditionalExpression::Tag{}));
    add(ExpressionKey(ast::ArrayReadExpression::Tag{}));
    for (ast::BinaryExpression::Op op : enumRange<ast::BinaryExpression::Op>())
      add(ExpressionKey(op));
    for (ast::UnaryExpression::Op op : enumRange<ast::UnaryExpression::Op>())
      add(ExpressionKey(op));
    add(StatementKey(ast::ArrayAssignmentStatement::Tag{}));
    add(StatementKey(ast::StructuredForStatement::Tag{}));
    return result;
  }();

  auto iter = keysByName.find(name);
  if (iter == keysByName.end())
    return nullptr;

  return &iter->second;
}

void ASTStatistic::update(const ast::Function &function) {
  NodeCounter(counts).visit(function);
  numSamples++;
}

void ASTStatistic::merge(const ASTStatistic &rhs) {
  numSamples += rhs.numSamples;
  for (const auto &[key, count] : rhs.counts)
    counts[key] += count;
}

/// Returns the average number of occurrences per sample of a node kind seen
/// 'count' times across 'numSamples' samples.
static double getAverage(std::size_t count, std::size_t numSamples) {
  return numSamples == 0 ? 0.0 : static_cast<double>(count) / numSamples;
}

void ASTStatistic::print(llvm::raw_ostream &os) const {
  os << CATEGORY << " (averaged over " << numSamples << " samples):\n";

  // Build a "<name>: <avg>" cell for every entry, tracking the widest one so
  // the columns can be aligned.
  std::vector<std::string> cells;
  cells.reserve(counts.size());
  std::size_t width = 0;
  for (const auto &[key, count] : counts) {
    double average = getAverage(count, numSamples);
    std::string &cell = cells.emplace_back();
    llvm::raw_string_ostream(cell)
        << getName(key) << ": " << llvm::format("%.2f", average);
    width = std::max(width, cell.size());
  }

  // Lay the cells out in a fixed number of columns.
  for (std::size_t i = 0; i < cells.size(); i++) {
    constexpr std::size_t numColumns = 3;
    bool lastInRow = i % numColumns == numColumns - 1 || i == cells.size() - 1;

    // Indent for the table.
    if (i % numColumns == 0)
      os << "  ";

    if (lastInRow)
      os << cells[i] << '\n';
    else
      os << llvm::left_justify(cells[i], width) << "  ";
  }
}

llvm::json::Value ASTStatistic::toJSON() const {
  llvm::json::Object averages;
  for (const auto &[key, count] : counts)
    averages[getName(key)] = getAverage(count, numSamples);

  return llvm::json::Object{
      {"numSamples", numSamples},
      {"averages", std::move(averages)},
  };
}

bool ASTStatistic::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  std::map<std::string, double> averages;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map("numSamples", numSamples) ||
      !mapper.map("averages", averages))
    return false;

  counts.clear();
  for (const auto &[name, average] : averages) {
    const NodeKey *key = lookupKey(name);
    if (!key) {
      path.field("averages").field(name).report("unknown AST node kind");
      return false;
    }

    // Recovers the count 'getAverage' divided by 'numSamples'. Both the
    // division and this multiplication are exact to within half a count for any
    // number of samples a fuzzing run can plausibly reach, as a double carries
    // 53 bits of mantissa.
    counts[*key] = std::llround(average * numSamples);
  }
  return true;
}
