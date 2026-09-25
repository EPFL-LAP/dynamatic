#ifndef DYNAMATIC_HLS_FUZZER_VISITOR_TYPE_SYSTEM
#define DYNAMATIC_HLS_FUZZER_VISITOR_TYPE_SYSTEM

#include "TemplateTypeSystem.h"
#include "TypeSystem.h"

namespace dynamatic::gen {

/// Convenient base class for any type systems that are just "counters".
/// These type systems:
/// * Do not care about the order that AST nodes are generated.
/// * Have a monotonic 'TypingContext' which only ever increases/decreases.
///
/// One property of this type system is that the most recent transfer function
/// called is guaranteed to receive the maximum/minimum 'TypingContext'
/// instance.
/// This makes the type system especially useful to implement counters and
/// similar.
///
/// The 'TypingContext' is required to contain a method with the signature:
/// 'TypingContext merge(const TypingContext& rhs) const' which can be used
/// to calculate the current maximum/minimum of all contexts generated so far.
template <typename TypingContext, typename Self>
class CounterTypeSystem : public TemplateTypeSystem<TypingContext, Self> {

public:
  /// A counter's default transfer function merges all present contexts, so
  /// whatever state a subelement's subtree accumulated flows onwards no
  /// matter in which order the generator walks the AST.
  template <typename ASTNode>
  static auto defaultTransferFn() {
    return getMergingTransferFn<ASTNode>();
  }

  /// Same for the default output transfer function: the node's output is the
  /// merge of its input and all its subelements' outputs.
  template <typename ASTNode>
  static auto defaultOutputTransferFn() {
    return getMergingOutputTransferFn<ASTNode>();
  }

protected:
  /// Returns a 'TransferFn' which merges all present contexts of 'indices' and
  /// the input context.
  template <typename ASTNode, std::size_t... indices>
  static auto getMergingTransferFn(std::index_sequence<indices...>) {
    return TransferFn<TypingContext, ASTNode, INPUT_DEPENDENCY,
                      weak(indices)...>(
        [](TypingContext result, auto &&...args) -> TypingContext {
          foreachInTuples(
              [&](const auto &element) {
                // Skip over the arguments that aren't the typing context such
                // as the ASTNodes passed alongside the contexts.
                if constexpr (std::is_same_v<std::decay_t<decltype(element)>,
                                             const TypingContext *>) {
                  if (!element)
                    return;
                  result = result.merge(*element);
                }
              },
              std::forward_as_tuple(std::forward<decltype(args)>(args)...));
          return result;
        });
  }

  /// Returns a 'TransferFn' which merges all present contexts including
  /// the input context.
  template <typename ASTNode>
  static auto getMergingTransferFn() {
    return getMergingTransferFn<ASTNode>(
        std::make_index_sequence<
            std::tuple_size_v<typename ASTNode::SubElements>>{});
  }

  /// Returns a 'OutputTransferFn' which merges all present contexts of
  /// 'indices' and the input context.
  template <typename ASTNode, std::size_t... indices>
  static auto getMergingOutputTransferFn(std::index_sequence<indices...>) {
    return OutputTransferFn<TypingContext, ASTNode, INPUT_DEPENDENCY,
                            indices...>(
        [](const ASTNode &, TypingContext result,
           const auto &...args) -> TypingContext {
          foreachInTuples(
              [&](const auto &element) {
                if constexpr (std::is_same_v<std::decay_t<decltype(element)>,
                                             TypingContext>) {
                  result = result.merge(element);
                }
              },
              std::forward_as_tuple(std::forward<decltype(args)>(args)...));
          return result;
        });
  }

  /// Returns a 'OutputTransferFn' which merges all present contexts including
  /// the input context.
  template <typename ASTNode>
  static auto getMergingOutputTransferFn() {
    return getMergingOutputTransferFn<ASTNode>(
        std::make_index_sequence<
            std::tuple_size_v<typename ASTNode::SubElements>>{});
  }
};

} // namespace dynamatic::gen

#endif
