#ifndef DYNAMATIC_HLS_FUZZER_RANDOMLY
#define DYNAMATIC_HLS_FUZZER_RANDOMLY

#include <cassert>
#include <random>
#include <unordered_set>

namespace dynamatic {

/// Class used to encapsulate all "random-ness" in the program.
/// Internally, a pseudo-random number generator is used such that values
/// returned are fully deterministic for a given seed.
class Randomly {
public:
  explicit Randomly(std::uint32_t seed) : generator(seed) {}

  /// Returns a random element from 'range'.
  /// Mustn't be empty.
  template <class Range>
  decltype(auto) fromRange(Range &&range) {
    size_t size = std::distance(range.begin(), range.end());
    assert(size != 0 && "cannot return element from empty range");
    auto index = getInteger<size_t>(0, size - 1);
    return *std::next(range.begin(), index);
  }

  /// Returns a random bool.
  bool getBool() {
    return std::uniform_int_distribution<uint32_t>(0, 1)(generator);
  }

  // Returns an "interesting" string.
  std::string getInterestingString() {
    const std::string charset = "abcdefghijklmnopqrstuvwxyz"
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "0123456789"
                                "!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\ \t";

    std::string result;
    size_t length = getInteger(0, 60);
    for (size_t i = 0; i < length; ++i) {
      result += charset[getInteger<size_t>(0, charset.size() - 1)];
    }
    return result;
  }

  /// Returns a random integer of type 'T'.
  template <class T>
  T getInteger() {
    return getInteger(std::numeric_limits<T>::min(),
                      std::numeric_limits<T>::max());
  }

  /// Returns a random integer of type 'T' in the range '[from, incTo]'.
  template <class T>
  T getInteger(T from, T incTo) {
    return std::uniform_int_distribution<T>(from, incTo)(generator);
  }

  /// Returns an "interesting" random integer of type 'T'.
  /// Interesting integers may with a certain chance return a boundary value or
  /// use some other kind of special cases that are not just a uniform
  /// distribution.
  template <class T = int64_t>
  T getInterestingInteger() {
    if (getSmallProbabilityBool()) {
      if constexpr (std::is_signed_v<T>)
        return fromRange(
            std::initializer_list<T>{-1, std::numeric_limits<T>::max(),
                                     std::numeric_limits<T>::min(), 1, 0});
      else
        return fromRange(
            std::initializer_list<T>{std::numeric_limits<T>::max(), 1, 0});
    }

    return getInteger<T>();
  }

  /// Returns an "interesting" random float. This may return a boundary value
  /// with a certain chance.
  float getInterestingFloat() {
    if (getSmallProbabilityBool())
      return fromRange(std::initializer_list<float>{
          0,
          -0.0f,
          std::numeric_limits<float>::max(),
          -std::numeric_limits<float>::max(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity(),
      });

    return std::uniform_real_distribution<float>()(generator);
  }

  /// Returns an "interesting" random double. This may return a boundary value
  /// with a certain chance.
  double getInterestingDouble() {
    if (getSmallProbabilityBool())
      return fromRange(std::initializer_list<double>{
          0,
          -0.0,
          std::numeric_limits<double>::max(),
          -std::numeric_limits<double>::max(),
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
      });

    return std::uniform_real_distribution<double>()(generator);
  }

  /// Returns a bool that is with a low probability true.
  bool getSmallProbabilityBool() {
    return std::uniform_int_distribution<uint32_t>(0, 99)(generator) == 0;
  }

  bool getRatherLowProbabilityBool() {
    return std::uniform_int_distribution<uint32_t>(0, 9)(generator) == 0;
  }

  /// Returns a random positive number that is "small".
  size_t getSmallNumber() {
    return static_cast<size_t>(std::abs(gaussian(generator)) * 2.0);
  }

  /// Returns a random enum value of the enum 'Enum'.
  /// 'Enum' must be made up of consecutive fields starting at 0 (the default in
  /// C++) and must have a special value called 'MAX_VALUE' which is equal to
  /// the last enum value.
  template <class Enum>
  Enum fromEnum() {
    static_assert(std::is_enum_v<Enum>);
    return static_cast<Enum>(
        getInteger<size_t>(0, static_cast<size_t>(Enum::MAX_VALUE)));
  }

  /// Returns a vector of a random subset of elements in 'range'.
  /// The returned vector is guaranteed to not be empty nor to have an element
  /// at a specific index of 'range' appear twice.
  /// 'range' must not be empty.
  template <class Range>
  auto getNonEmptySubset(Range &&range) {
    std::vector<typename std::decay_t<Range>::value_type> result;
    size_t size = std::distance(range.begin(), range.end());
    assert(size > 0 && "range mustn't be empty");

    auto numResult = getInteger<size_t>(1, size);
    result.reserve(numResult);

    // Keep track of seen indices to avoid duplicates.
    std::unordered_set<size_t> seen;
    while (result.size() < numResult) {
      auto index = getInteger<size_t>(0, size - 1);
      if (!seen.insert(index).second)
        continue;

      result.push_back(*std::next(range.begin(), index));
    }

    return result;
  }

private:
  std::minstd_rand generator;
  std::normal_distribution<> gaussian{0, 1};
};

} // namespace dynamatic

#endif
