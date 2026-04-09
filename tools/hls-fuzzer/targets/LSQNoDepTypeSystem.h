#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_LSQNODEPTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_TARGETS_LSQNODEPTYPESYSTEM

#include "DynamaticTypeSystem.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"

namespace dynamatic::gen {

struct LSQNoDepContext {
  /// The array parameter + index that is being written to in the current
  /// statement, or empty if not.
  /// Correctness relies on the fact that the indexing expression cannot have
  /// side effects beyond reading array values.
  std::optional<std::pair<ast::ArrayParameter *, ast::Expression>>
      elementWritten = std::nullopt;
};

namespace detail {

/// Subtype system used in conjunction with the Dynamatic type system.
/// 'check*' methods should be implemented here for composability.
class LSQNoDepTypeSystemInner final
    : public TypeSystem<LSQNoDepContext, LSQNoDepTypeSystemInner> {
public:
  using TypeSystem::TypeSystem;

  static std::optional<ConclusionOf<ast::ReturnType>>
  checkReturnType(const ast::ReturnType &returnType, const LSQNoDepContext &) {
    // Force a void return function.
    if (returnType != ast::ReturnType{ast::VoidType{}})
      return std::nullopt;

    return ConclusionOf<ast::ReturnType>{};
  }
};

} // namespace detail

/// Type system used to generate code that requires no LSQ to enforce ordering
/// constraints of memory.
/// Concretely this means that:
/// * For any Write-after-Read (WAR), the write operation must be data dependent
///   on the read, guaranteed not to alias OR have a control dependency on the
///   read.
/// * For any Write-after-Write (WAW) or Read-after-Write (RAW) the operations
///   must not alias.
///
/// The current implementation only ever generates a single WAR construct where
/// the write is data dependent or control dependent on the read.
class LSQNoDepTypeSystem final
    : public ConjunctionTypeSystem<LSQNoDepTypeSystem,
                                   detail::LSQNoDepTypeSystemInner,
                                   DynamaticTypeSystem> {
public:
  using ConjunctionTypeSystem::ConjunctionTypeSystem;

  std::optional<ast::ArrayReadExpression>
  generateArrayReadExpression(const Context &context,
                              GenerateCallback<ast::ArrayParameter, Context>,
                              GenerateCallback<ast::Expression, Context>);

  std::optional<ast::ArrayAssignmentStatement> generateArrayAssignmentStatement(
      const Context &context,
      GenerateCallback<ast::ArrayParameter, Context> generateArrayParameter,
      GenerateCallback<ast::Expression, Context> generateExpression);

  static std::vector<ast::Statement> generateStatementList(
      const Context &context,
      GenerateCallback<ast::Statement, Context> generateStatement) {
    // Generate exactly one statement for now.
    // This makes it such that we do not have to reason between array-accesses
    // in other statements.
    std::optional<ast::Statement> statement = generateStatement(context);
    assert(statement && "it must always be possible to generate a statement");
    return {std::move(*statement)};
  }
};

} // namespace dynamatic::gen

#endif
