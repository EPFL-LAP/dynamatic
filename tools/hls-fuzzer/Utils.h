#ifndef DYNAMATIC_HLS_FUZZER_UTILS
#define DYNAMATIC_HLS_FUZZER_UTILS

#include <tuple>

namespace dynamatic {

//===----------------------------------------------------------------------===//
//
// Utility methods that operate on tuples.
// These are different from existing range based methods in LLVM or the C++
// standard library in that they need to be able to handle the heterogeneity of
// tuples, making existing methods impossible to use with tuples.
//
//===----------------------------------------------------------------------===//

/// Given a an 'std::index_sequence' filled with the values 'is', constructs
/// an 'std::tuple' of 'std::integral_constant's with the values in 'is'.
///
/// This allows passing the tuples to functions operating on tuples such as
/// 'std::apply' or others. The indices can still be used in constant
/// evaluation by querying their type and converting it to an integer.
template <std::size_t... is>
constexpr auto getTupleOfIndices(std::index_sequence<is...>) {
  return std::tuple{std::integral_constant<std::size_t, is>{}...};
}

/// Applies the function 'f' to all elements in the tuple 'first' and 'others'
/// and passes all results as one parameter pack to 'constructor'.
///
/// 'first' and elements in 'others' must be tuples of the same length.
/// Example:
///   std::array<float, 3> result = mapTuplesInto([](auto&&...values) {
///     return std::array{values...};
///   }, [](std::size_t i, float f) {
///     return i * f;
///   }, std::tuple(5, 3, 8), std::tuple(3.0, 2.0, 6.0));
///   // result == {15.0, 6.0, 48.0}
template <typename Constructor, typename First, typename... Others, typename F>
decltype(auto) mapTuplesInto(Constructor &&constructor, F &&f, First &&first,
                             Others &&...others) {
  return std::apply(
      [&](auto &&...indices) {
        return std::forward<Constructor>(constructor)(
            [&](auto index) -> decltype(auto) {
              return f(
                  std::get<decltype(index){}>(std::forward<First>(first)),
                  std::get<decltype(index){}>(std::forward<Others>(others))...);
            }(indices)...);
      },
      getTupleOfIndices(
          std::make_index_sequence<std::tuple_size_v<std::decay_t<First>>>{}));
}

/// Like 'mapTuplesInto' but the constructor is hardcoded to produce an array.
template <typename First, typename... Others, typename F>
decltype(auto) mapTuplesIntoArray(F &&f, First &&first, Others &&...others) {
  return mapTuplesInto(
      [](auto &&...args) {
        return std::array{std::forward<decltype(args)>(args)...};
      },
      std::forward<F>(f), std::forward<First>(first),
      std::forward<Others>(others)...);
}

/// Like 'mapTuplesInto' but the constructor is hardcoded to produce a tuple.
template <typename First, typename... Others, typename F>
decltype(auto) mapTuples(F &&f, First &&first, Others &&...others) {
  return mapTuplesInto(
      [](auto &&...args) {
        return std::make_tuple(std::forward<decltype(args)>(args)...);
      },
      std::forward<F>(f), std::forward<First>(first),
      std::forward<Others>(others)...);
}

/// Like 'mapTuplesInto' the function 'f' additionally receives an index
/// parameter ranging from 0 to the length of the tuples as the first parameter.
/// The index parameter is of type 'std::integral_constant' and can therefore be
/// used in constant evaluation.
///
/// Example:
///   std::tuple<float, std::size_t, int> result =
///     enumerateTuplesInto([](auto&&...values) {
///       return std::tuple{values...};
///     }, [](auto indexT, auto f) {
///       constexpr std::size_t i = decltype(indexT){};
///       return i * f;
///     }, std::tuple(3.0f, 2ull, 6));
///   // result == {0.0f, 2ull, 18}
template <typename Constructor, typename First, typename... Others, typename F>
decltype(auto) enumerateTuplesInto(Constructor &&constructor, F &&f,
                                   First &&first, Others &&...others) {
  return mapTuplesInto(
      std::forward<Constructor>(constructor), std::forward<F>(f),
      getTupleOfIndices(
          std::make_index_sequence<std::tuple_size_v<std::decay_t<First>>>{}),
      std::forward<First>(first), std::forward<Others>(others)...);
}

/// Like 'enumerateTuplesInto' but the constructor is hardcoded to produce a
/// tuple.
template <typename First, typename... Others, typename F>
decltype(auto) enumerateTuples(F &&f, First &&first, Others &&...others) {
  return enumerateTuplesInto(
      [](auto &&...args) {
        return std::make_tuple(std::forward<decltype(args)>(args)...);
      },
      std::forward<F>(f), std::forward<First>(first),
      std::forward<Others>(others)...);
}

/// Applies the function 'f' to all elements in the tuples 'first' and 'others'
/// at the same time (like 'zip' in many programming languages).
/// 'first' and 'others' are required to be of the same length.
/// 'f' is not expected to return any value.
template <typename First, typename... Others, typename F>
void foreachInTuples(F &&f, First &&first, Others &&...others) {
  // Discard any results of 'f' and make the constructor a noop.
  mapTuplesInto([](auto &&...) { return 0; },
                [&](auto &&...args) {
                  f(std::forward<decltype(args)>(args)...);
                  return 0;
                },
                std::forward<First>(first), std::forward<Others>(others)...);
}

} // namespace dynamatic

#endif
