#ifndef HLS_FUZZER_TARGETS_TERMINATIONTYPESYSTEM
#define HLS_FUZZER_TARGETS_TERMINATIONTYPESYSTEM

#include "BitwidthTypeSystem.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/OptionalTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"
#include "hls-fuzzer/VariableTrackingTypeSystem.h"

namespace dynamatic::gen {

/// Implementation details of 'TerminationTypeSystem' that are not part of its
/// public interface.
namespace detail {

/// Typing context for the 'IterationVariableTypeSystem'.
///
/// The variables it tracks are the iteration variables of all loops generated
/// so far, including ones that have already ended: the generator never reuses
/// a variable name and never offers a variable that is out of scope, so the
/// stale entries can never be acted upon.
struct IterationVariableTypingContext : VariableTrackingTypingContext<NoValue> {
  /// True if the current expression is the target of a scalar assignment (i.e.,
  /// its left-hand side), meaning a value is about to be written to it.
  bool isWriteTarget = false;

  /// Returns true if 'name' refers to a loop iteration variable.
  bool isIterationVariable(llvm::StringRef name) const {
    return isTracked(name);
  }
};

/// Sub type system whose sole responsibility is to disallow writing to loop
/// iteration variables.
///
/// Writing to the iteration variable of an enclosing loop (e.g., resetting it
/// to zero on every iteration) may prevent the loop from ever terminating.
/// Writing to any other variable (a function parameter or a freshly introduced
/// one) is harmless with respect to termination and therefore left untouched.
class IterationVariableTypeSystem
    : public VariableTrackingTypeSystemBase<IterationVariableTypingContext,
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
        /*iteration variable=*/defaultTransferFn<ast::StructuredForStatement>(),
        // The bounds need no handling of their own: writes only ever happen
        // through assignment statements, which the bounds being
        // expressions cannot contain.
        /*start=*/defaultTransferFn<ast::StructuredForStatement>(),
        /*end=*/defaultTransferFn<ast::StructuredForStatement>(),
        /*step=*/defaultTransferFn<ast::StructuredForStatement>(),
        /*body=*/
        // Note that unlike everywhere else this is a non-weak dependency: the
        // iteration variable has to be known before the body is generated, as
        // the whole point is to keep the body from writing to it.
        defaultTransferFn<ast::StructuredForStatement>()
            .wrap<ast::StructuredForStatement::ITER_VARIABLE>(
                [this](llvm::function_ref<IterationVariableTypingContext()>
                           wrapped,
                       const IterationVariableTypingContext &,
                       const std::string &iterVariable) {
                  return track(wrapped(), iterVariable, {});
                }),
        /*output=*/defaultOutputTransferFn<ast::StructuredForStatement>(),
    };
  }

  TransferFnArray<ast::ScalarAssignmentStatement>
  getScalarAssignmentStatementTransferFns() override {
    return {
        /*target=*/
        defaultTransferFn<ast::ScalarAssignmentStatement>().wrap(
            [](llvm::function_ref<IterationVariableTypingContext()> wrapped) {
              // Mark the target such that iteration variables are rejected
              // while generating it.
              IterationVariableTypingContext context = wrapped();
              context.isWriteTarget = true;
              return context;
            }),
        /*value=*/defaultTransferFn<ast::ScalarAssignmentStatement>(),
        /*output=*/defaultOutputTransferFn<ast::ScalarAssignmentStatement>(),
    };
  }
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
