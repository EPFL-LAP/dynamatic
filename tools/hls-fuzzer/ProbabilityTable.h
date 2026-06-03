#ifndef HLS_FUZZER_PROBABILITY_TABLE
#define HLS_FUZZER_PROBABILITY_TABLE

#include <map>

namespace dynamatic {

/// Table representing relative probabilities of a given key to appear earlier
/// in a shuffle-range operation. See the 'Randomly' class for such a shuffle
/// operation.
///
/// Conceptually, this is a 'Key' to unsigned integer mapping where the weight
/// of an element corresponds to its probability relative to all other
/// not-yet-selected elements to occur earlier.
/// This means an element with weight 2 is twice as likely to be selected before
/// an element of weight 1.
///
/// Weight 0 can be used to disable an element being in the shuffle result.
/// 'Key' must be a type supporting 'operator<'.
///
/// Elements not explicitly part of the probability table are assumed to have
/// weight 1 ('FALLBACK_WEIGHT').
template <typename Key>
class ProbabilityTable {
public:
  /// Constructs an empty probability table.
  ProbabilityTable() = default;

  /// Initializes a probability table from the given initializer list.
  ProbabilityTable(std::initializer_list<std::pair<Key, std::size_t>> list)
      : keyToWeight(std::move(list)) {}

  /// Initializes a probability table from the given range of
  /// 'std::pair<Key, std::size_t>'s.
  template <typename Range,
            std::enable_if_t<!std::is_same_v<std::decay_t<Range>,
                                             ProbabilityTable>> * = nullptr>
  explicit ProbabilityTable(Range &&range)
      : keyToWeight(range.begin(), range.end()) {}

  constexpr static std::size_t FALLBACK_WEIGHT = 1;

  /// Returns a reference to the weight of the given key. Inserts the key with
  /// the fallback weight if it did not exist previously.
  std::size_t &operator[](const Key &key) {
    return keyToWeight.emplace(key, FALLBACK_WEIGHT).first->second;
  }

  /// Returns the weight for the given key.
  /// Returns the fallback weight if no entry exists for the given key.
  std::size_t getWeight(const Key &key) const {
    auto result = keyToWeight.find(key);
    if (result != keyToWeight.end())
      return result->second;

    return FALLBACK_WEIGHT;
  }

  /// Combines the probability table 'other' into '*this'.
  /// This is done by multiplying the weights.
  void inplaceMerge(const ProbabilityTable &other) {
    for (auto &iter : other.keyToWeight)
      (*this)[iter.first] *= iter.second;
  }

private:
  std::map<Key, std::size_t> keyToWeight;
};
} // namespace dynamatic

#endif
