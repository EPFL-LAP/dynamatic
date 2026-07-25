#ifndef HLS_FUZZER_TARGETS_TERMINATIONTYPESYSTEM
#define HLS_FUZZER_TARGETS_TERMINATIONTYPESYSTEM

#include "BitwidthTypeSystem.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/OptionalTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"

#include "llvm/ADT/ImmutableSet.h"

#include <memory>

namespace dynamatic::gen {

/// Implementation details of 'TerminationTypeSystem' that are not part of its
/// public interface.
namespace detail {

/// 'llvm::ImmutableSet' traits for 'std::string' elements.
///
/// The library's default trait profiles elements via a 'Profile' member
/// function, which 'std::string' does not have; this instead profiles
/// elements by their string content.
struct IterationVariableSetInfo : llvm::ImutContainerInfo<std::string> {
  static void Profile(llvm::FoldingSetNodeID &id, const std::string &value) {
    id.AddString(value);
  }
};

using VariableSet = llvm::ImmutableSet<std::string, IterationVariableSetInfo>;

/// Typing context for the 'IterationVariableTypeSystem'.
struct IterationVariableTypingContext {
  /// The iteration variables of all enclosing loops that are currently in
  /// scope.
  /// Note that the generator never reassigns a variable naming, meaning we do
  /// not need to care about C scope handling here.
  ///
  /// The set is backed by an immutable (persistent) data structure since
  /// contexts are copied frequently during generation: copying it is thus
  /// cheap, while entering a new loop only allocates the nodes needed to
  /// represent the single newly added variable, structurally sharing
  /// everything else with the previous set.
  VariableSet iterationVariables{nullptr};

  /// True if the current expression is the target of a scalar assignment (i.e.,
  /// its left-hand side), meaning a value is about to be written to it.
  bool isWriteTarget = false;

  /// Returns true if 'name' refers to an in-scope loop iteration variable.
  bool isIterationVariable(llvm::StringRef name) const {
    return iterationVariables.contains(name.str());
  }
};

/// Sub type system whose sole responsibility is to disallow writing to loop
/// iteration variables.
///
/// Writing to the iteration variable of an enclosing loop (e.g., resetting it
/// to zero on every iteration) may prevent the loop from ever terminating.
/// Writing to any other variable (a function parameter or a freshly introduced
/// one) is harmless with respect to termination and therefore left untouched.
class IterationVariableTypeSystem final
    : public TypeSystem<IterationVariableTypingContext,
                        IterationVariableTypeSystem> {
public:
  /// Discards an existing variable as an assignment target if it is a loop
  /// iteration variable.
  static bool discardExistingScalarParameter(
      const ast::ScalarParameter &parameter,
      const IterationVariableTypingContext &context) {
    return context.isWriteTarget &&
           context.isIterationVariable(parameter.getName());
  }

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override {
    return {
        /*iteration variable=*/copyFromInput<ast::StructuredForStatement>(),
        /*start=*/copyFromInput<ast::StructuredForStatement>(),
        /*end=*/copyFromInput<ast::StructuredForStatement>(),
        /*step=*/copyFromInput<ast::StructuredForStatement>(),
        /*body=*/
        TransferFn<ast::StructuredForStatement, INPUT_DEPENDENCY,
                   ast::StructuredForStatement::ITER_VARIABLE>(
            [this](IterationVariableTypingContext context,
                   const IterationVariableTypingContext &,
                   const std::string &iterVariable) {
              // Add the loop's iteration variable to the set of variables that
              // must not be written to within the body.
              context.iterationVariables =
                  factory->add(context.iterationVariables, iterVariable);
              return context;
            }),
        /*output=*/copyInputToOutput<ast::StructuredForStatement>(),
    };
  }

  TransferFnArray<ast::ScalarAssignmentStatement>
  getScalarAssignmentStatementTransferFns() override {
    return {
        /*target=*/
        TransferFn<ast::ScalarAssignmentStatement, INPUT_DEPENDENCY>(
            [](IterationVariableTypingContext context) {
              // Mark the target such that iteration variables are rejected
              // while generating it.
              context.isWriteTarget = true;
              return context;
            }),
        /*value=*/copyFromInput<ast::ScalarAssignmentStatement>(),
        /*output=*/copyInputToOutput<ast::ScalarAssignmentStatement>(),
    };
  }

private:
  /// Factory backing every set produced for
  /// 'IterationVariableTypingContext::iterationVariables'.
  /// Held behind a 'unique_ptr' as the type system itself must stay movable for
  /// conjunction.
  std::unique_ptr<VariableSet::Factory> factory =
      std::make_unique<VariableSet::Factory>();
};

} // namespace detail

/// Type system whose goal is to guarantee that generated programs always
/// terminate (i.e., contain no infinite loops).
///
/// It combines two independent sub type systems:
/// * An optionally enabled 'BitwidthTypeSystem' which is used to bound the
///   number of iterations a loop may run by restricting its bounds to small
///   bitwidths.
/// * An 'IterationVariableTypeSystem' which forbids writing to loop iteration
///   variables.
///
/// Together they ensure that a loop's iteration variable is monotonically
/// advanced towards a bounded end value, guaranteeing termination.
class TerminationTypeSystem final
    : public ConjunctionTypeSystemBase<TerminationTypeSystem,
                                       OptionalTypeSystem<BitwidthTypeSystem>,
                                       detail::IterationVariableTypeSystem> {
public:
  explicit TerminationTypeSystem(Randomly &random)
      : ConjunctionTypeSystemBase(
            // Upper bound on what bitwidth can be generated in unrestricted
            // contexts. Only ever enabled for loop bounds (see below).
            BitwidthTypeSystem(64, random),
            detail::IterationVariableTypeSystem()) {}

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override;
};

} // namespace dynamatic::gen

#endif
