#ifndef HLS_FUZZER_CONJUNCTION_TYPE_SYSTEM
#define HLS_FUZZER_CONJUNCTION_TYPE_SYSTEM

#include "TypeSystem.h"
#include <tuple>

namespace dynamatic::gen {

/// Base class for creating type systems by combining multiple independent type
/// systems.
/// The derived class should be specified as the 'Self' parameter while the
/// subtype systems should be specified as 'SubTypeSystem' parameter.
///
/// The typing context of the type system is a tuple of all contexts of the
/// subtype systems. All 'check*' methods have a default implementation which
/// calls all 'check*' methods of all subtype systems.
/// If any of the methods return an empty optional, the AST node is discarded.
///
/// Note that 'generate*' methods of subtype systems do *not* compose and are
/// completely ignored.
/// These have to be reimplemented in the class deriving from
/// 'ConjunctionTypeSystem'.
template <typename Self, class... SubTypeSystem>
class ConjunctionTypeSystem
    : public TypeSystem<std::tuple<typename SubTypeSystem::Context...>, Self> {
public:
  using Base = TypeSystem<std::tuple<typename SubTypeSystem::Context...>, Self>;
  using Context = typename Base::Context;

  explicit ConjunctionTypeSystem(Randomly &randomly)
      : Base(randomly), typeSystems(SubTypeSystem(randomly)...) {}

  // Default implementations of all 'check*' methods follow.

  ConclusionOf<ast::Function, Context> checkFunction(const Context &context) {
    // Functions cannot be discarded hence the optional will never be empty.
    return *combineChecks<ast::Function>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkFunction(context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::BinaryExpression, Context>>
  checkBinaryExpression(ast::BinaryExpression::Op op, const Context &context) {
    return combineChecks<ast::BinaryExpression>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkBinaryExpression(op, context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::CastExpression, Context>>
  checkCastExpression(const Context &context) {
    return combineChecks<ast::CastExpression>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkCastExpression(context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ConditionalExpression, Context>>
  checkConditionalExpression(const Context &context) {
    return combineChecks<ast::ConditionalExpression>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkConditionalExpression(context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::Variable, Context>>
  checkVariable(const Context &context) {
    return combineChecks<ast::Variable>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkVariable(context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ScalarParameter, Context>>
  checkScalarParameter(const ast::ScalarParameter &scalarParameter,
                       const Context &context) {
    return combineChecks<ast::ScalarParameter>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkScalarParameter(scalarParameter, context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ArrayParameter, Context>>
  checkArrayParameter(const ast::ArrayParameter &arrayParameter,
                      const Context &context) {
    return combineChecks<ast::ArrayParameter>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkArrayParameter(arrayParameter, context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ScalarType, Context>>
  checkScalarType(const ast::ScalarType &scalarType, const Context &context) {
    return combineChecks<ast::ScalarType>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkScalarType(scalarType, context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ReturnType, Context>>
  checkReturnType(const ast::ReturnType &returnType, const Context &context) {
    return combineChecks<ast::ScalarType>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkReturnType(returnType, context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ArrayReadExpression, Context>>
  checkArrayReadExpression(const Context &context) {
    return combineChecks<ast::ArrayReadExpression>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkArrayReadExpression(context);
        },
        context);
  }

  std::optional<ConclusionOf<ast::ArrayAssignmentStatement, Context>>
  checkArrayAssignmentStatement(const Context &context) {
    return combineChecks<ast::ArrayAssignmentStatement>(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.checkArrayAssignmentStatement(context);
        },
        context);
  }

protected:
  /// Convenience template method which combines the result of calling
  /// 'checkCallback' with a specific subtype system and corresponding context
  /// with 'context' as input context.
  /// 'checkCallback' should be generic and allow any type as the first
  /// typesystem parameter and any context as the second parameter.
  /// 'ASTNode' is the node being type checked and whose conclusion type should
  /// be returned.
  ///
  /// Returns an empty optional if any of the subtype systems returns an empty
  /// optional.
  template <typename ASTNode, typename F>
  std::optional<ConclusionOf<ASTNode, Context>>
  combineChecks(F &&checkCallback, const Context &context) {
    // Call the check method on every type system.
    // Effectively does a 'zip' operation on both the 'typeSystems' tuple and
    // the 'context' tuple.
    auto subChecks = std::apply(
        [&](auto... indices) {
          return std::make_tuple(std::optional{
              checkCallback(std::get<decltype(indices)::value>(typeSystems),
                            std::get<decltype(indices)::value>(context))}...);
        },
        getIndicesTuple());

    // Check whether any of them are empty.
    if (std::apply(
            [](auto &&...optionals) {
              return (false || ... || !optionals.has_value());
            },
            subChecks))
      return std::nullopt;

    if constexpr (std::is_same_v<ConclusionOf<ASTNode, Context>, Context>) {
      // Conclusion type is just 'Context'. Construct it by dereferencing the
      // optionals.
      return std::apply(
          [&](auto &&...subChecks) {
            return std::make_tuple(std::move(*subChecks)...);
          },
          std::move(subChecks));
    } else {
      // Conclusion type is a tuple of 'Context's.
      // We need to combine all the sub-conclusion types by creating tuples
      // (Context's) for every element of the conclusion type.
      return std::apply(
          [&](auto &&...subChecks) {
            return std::apply(
                [&](auto... indices) {
                  return ConclusionOf<ASTNode, Context>{[&](auto index) {
                    // For the element at the given index in the conclusion
                    // type, create a Context from all the sub-conclusion types.
                    return std::make_tuple(
                        get<decltype(index)::value>(std::move(*subChecks))...);
                  }(indices)...};
                },
                // Indices for every element of the conclusion type.
                getIndicesTuple(
                    std::make_index_sequence<
                        std::tuple_size_v<ConclusionOf<ASTNode, Context>>>{}));
          },
          std::move(subChecks));
    }
  }

  std::tuple<SubTypeSystem...> typeSystems;

private:
  template <std::size_t... is>
  constexpr auto getIndicesTuple(std::index_sequence<is...>) {
    return std::tuple{std::integral_constant<std::size_t, is>{}...};
  }

  constexpr auto getIndicesTuple() {
    return getIndicesTuple(std::index_sequence_for<SubTypeSystem...>{});
  }
};

} // namespace dynamatic::gen

#endif
