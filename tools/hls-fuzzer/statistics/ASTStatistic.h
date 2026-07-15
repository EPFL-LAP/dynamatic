#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS_ASTSTATISTICS
#define DYNAMATIC_HLS_FUZZER_STATISTICS_ASTSTATISTICS

#include "hls-fuzzer/TypeSystem.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <map>
#include <variant>

namespace dynamatic {

/// Statistic gathering the average number of expressions and statements of each
/// kind across all sampled ASTs. Nodes are bucketed at the granularity of
/// 'AbstractTypeSystem::ExpressionKey' and 'AbstractTypeSystem::StatementKey',
/// meaning binary and unary expressions are distinguished by their operator.
class ASTStatistic {
public:
  /// Records the expressions and statements contained in 'function' as an
  /// additional sample.
  void update(const ast::Function &function);

  /// Merges the counts gathered by 'rhs' into this statistic.
  void merge(const ASTStatistic &rhs);

  void print(llvm::raw_ostream &os) const;

  /// Returns an object containing the number of sampled ASTs and the average
  /// number of occurrences of every node kind within them.
  llvm::json::Value toJSON() const;

  constexpr static llvm::StringRef CATEGORY = "AST Statistics";

private:
  using ExpressionKey = gen::AbstractTypeSystem::ExpressionKey;
  using StatementKey = gen::AbstractTypeSystem::StatementKey;
  /// Key identifying a tracked AST node: either an expression or a statement.
  using NodeKey = std::variant<ExpressionKey, StatementKey>;

  /// Number of ASTs (functions) that have been sampled.
  std::size_t numSamples = 0;

  /// Total number of nodes of each kind summed across all samples.
  std::map<NodeKey, std::size_t> counts;
};

} // namespace dynamatic
#endif
