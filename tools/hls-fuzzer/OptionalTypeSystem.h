#ifndef DYNAMATIC_HLS_FUZZER_OPTIONALTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_OPTIONALTYPESYSTEM

#include "TemplateTypeSystem.h"
#include "TypeSystem.h"
#include <optional>

namespace dynamatic::gen {

/// Wrapper around a 'SubTypeSystem' that allows one to selectively enable
/// the type system as needed.
/// To enable the type system the 'enableTypeSystemFor' function should be
/// used.
/// Otherwise, if the context is an empty optional, and the type system
/// therefore disabled, it acts identical to a 'NoopTypeSystem'.
/// It allows enabling the type system for specific sub-elements of an
/// 'ASTNode' by setting their input contexts to a given entryContext.
///
/// An 'OptionalTypeSystem' is therefore best used within a
/// 'ConjunctionTypeSystem' or similar, where another type systems logic can
/// enable sub-trees.
///
template <typename SubTypeSystem>
class OptionalTypeSystem final
    : public TemplateTypeSystem<std::optional<typename SubTypeSystem::Context>,
                                OptionalTypeSystem<SubTypeSystem>> {
public:
  using SubContext = typename SubTypeSystem::Context;
  using Base =
      TemplateTypeSystem<std::optional<SubContext>, OptionalTypeSystem>;
  using Context = typename Base::Context;
  /// 'TypeSystem' and its default implementations, which this class falls back
  /// to while disabled.
  using Default = typename Base::Base;

  /*implicit*/ OptionalTypeSystem(SubTypeSystem &&subTypeSystem)
      : subTypeSystem(std::move(subTypeSystem)) {}

  /// Enables the sub type system for specific 'subElementsToEnable' of
  /// 'ASTNode'.
  /// The input context given to these sub-elements is 'context'.
  /// It is the users responsibility to make sure that the given 'context'
  /// makes sense for the sub type system.
  template <std::size_t... subElementsToEnable, typename TransferFns>
  static TransferFns enableTypeSystemFor(TransferFns transferFnArray,
                                         const SubContext &context) {
    using ASTNode = TransferFnArrayASTNode<TransferFns>;
    // For the given sub-elements that should be enabled, overwrite their
    // transfer functions such that they now return the given 'context'.
    ((std::get<subElementsToEnable>(transferFnArray) =
          typename Base::template TransferFn<ASTNode>(context)),
     ...);
    return transferFnArray;
  }

  /// Every AST node is treated the same way: the transfer functions of the sub
  /// type system are wrapped so that they can be enabled and disabled.
  template <typename ASTNode, typename... Args>
  TransferFnArray<ASTNode> getTransferFnImpl(const Args &...args) {
    return wrapTransferFns(
        subTypeSystem.template getTransferFn<ASTNode>(args...));
  }

  /// While disabled, no terminal is ever discarded.
  template <typename ASTNode>
  bool discardTerminalImpl(const ASTNode &node, const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardTerminal(node, *context);
  }

  /// Defined on top of 'discardTerminalImpl' as the sub type system may replace
  /// the constant with another one, which a terminal cannot express.
  std::optional<ast::Constant> discardConstant(const ast::Constant &constant,
                                               const Context &context) {
    if (!context)
      return constant;
    return subTypeSystem.discardConstant(constant, *context);
  }

  /// While disabled, no non-terminal is ever discarded.
  template <typename ASTNode, typename... Args>
  bool discardNonTerminalImpl(const Context &context, const Args &...args) {
    if (!context)
      return false;
    return subTypeSystem.template discardNonTerminal<ASTNode>(*context,
                                                              args...);
  }

  /// While disabled, every probability table is the one 'TypeSystem' defaults
  /// to, i.e. one biasing nothing.
  template <typename Key>
  ProbabilityTable<Key> getProbabilityTableImpl(const Context &context) {
    if (!context)
      return Default::template getProbabilityTable<Key, Default>(context);

    return subTypeSystem.template getProbabilityTable<Key>(*context);
  }

private:
  /// Wraps around the existing transfer functions in 'array' to add the ability
  /// of enabling and disabling the sub type system.
  template <typename TransferFns>
  static TransferFns wrapTransferFns(TransferFns &&array) {
    using ASTNode = TransferFnArrayASTNode<TransferFns>;
    // Unwraps a context of an enabled subtree into the
    // 'SubTypeSystem::Context' the sub type system accepts.
    auto unwrap = [](const Context &context) -> const SubContext & {
      assert(context.has_value() &&
             "input context being enabled implies the entire sub-tree of "
             "contexts being enabled");
      return context.value();
    };

    return mapTuples(
        /*mappingFunction=*/
        [&](auto &&element) {
          using T = std::decay_t<decltype(element)>;
          if constexpr (std::is_same_v<T, OpaqueTransferFn<ASTNode>>) {
            return std::move(element).template wrap<Context, INPUT_DEPENDENCY>(
                [](llvm::function_ref<SubContext()> wrapped,
                   const Context &input) -> Context {
                  // If the input context is enabled, then all transfer
                  // functions of the sub type system are enabled.
                  if (!input)
                    return std::nullopt;

                  return wrapped();
                },
                unwrap);
          } else {
            return std::move(element).template wrap<Context, INPUT_DEPENDENCY>(
                [](llvm::function_ref<SubContext()> wrapped, const ASTNode &,
                   const Context &input) -> Context {
                  // If the input context is enabled, then all transfer
                  // functions of the sub type system must have returned an
                  // output context.
                  if (!input)
                    return std::nullopt;

                  return wrapped();
                },
                unwrap);
          }
        },
        /*tuple=*/std::move(array));
  }

  SubTypeSystem subTypeSystem;
};
} // namespace dynamatic::gen
#endif
