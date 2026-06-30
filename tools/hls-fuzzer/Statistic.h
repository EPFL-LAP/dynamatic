#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS
#define DYNAMATIC_HLS_FUZZER_STATISTICS

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace dynamatic {

/// "Base" class wrapper for all kinds of statistics.
/// Concrete statistics are expected to have the following interface:
/// * Copyable and moveable
/// * 'void merge(const ConcreteStatistic &other)' to merge multiple instances
///   into one.
/// * 'void print(llvm::raw_ostream &os) const' to display it to the user.
class Statistic {
public:
  template <typename ConcreteStatistic>
  explicit Statistic(std::string name, ConcreteStatistic &&statistic)
      : category(std::move(name)),
        value(std::make_unique<Derived<std::decay_t<ConcreteStatistic>>>(
            std::forward<ConcreteStatistic>(statistic))) {}

  ~Statistic() = default;

  Statistic(const Statistic &rhs)
      : category(rhs.category), value(rhs.value->copy()) {}

  Statistic &operator=(const Statistic &rhs) {
    if (this != &rhs) {
      this->~Statistic();
      new (this) Statistic(rhs);
    }
    return *this;
  }

  Statistic(Statistic &&rhs) noexcept
      : category(rhs.category), value(rhs.value->move()) {}

  Statistic &operator=(Statistic &&rhs) noexcept {
    if (this != &rhs) {
      this->~Statistic();
      new (this) Statistic(std::move(rhs));
    }
    return *this;
  }

  /// The name of the category this collection of statistics belongs to.
  llvm::StringRef getCategory() const { return category; }

  /// Merges the statistics of both 'this' and 'other' into 'this'.
  /// Callers must ensure the statistics are of the same category.
  void merge(const Statistic &other) {
    assert(category == other.category &&
           "expected only statistics of the same category to be mergeable");
    value->merge(*other.value);
  }

  void print(llvm::raw_ostream &os) const { value->print(os); }

private:
  struct Base {
    virtual ~Base() = default;

    virtual std::unique_ptr<Base> move() const = 0;

    virtual std::unique_ptr<Base> copy() const = 0;

    virtual void merge(const Base &other) = 0;

    virtual void print(llvm::raw_ostream &os) const = 0;
  };

  template <typename T>
  struct Derived final : Base {
    T data;

    explicit Derived(T &&data) : data(std::move(data)) {}
    explicit Derived(const T &data) : data(data) {}

    std::unique_ptr<Base> move() const noexcept override {
      return std::make_unique<Derived>(std::move(data));
    }

    std::unique_ptr<Base> copy() const override {
      return std::make_unique<Derived>(data);
    }

    void merge(const Base &other) override {
      data.merge(static_cast<const Derived &>(other).data);
    }

    void print(llvm::raw_ostream &os) const override { data.print(os); }
  };

  std::string category;
  std::unique_ptr<Base> value;
};

} // namespace dynamatic

#endif
