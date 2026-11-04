#ifndef DYNAMATIC_HLS_FUZZER_VARIABLE_TRACKING_TYPE_SYSTEM
#define DYNAMATIC_HLS_FUZZER_VARIABLE_TRACKING_TYPE_SYSTEM

#include "CounterTypeSystem.h"
#include "TypeSystem.h"

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace dynamatic::gen {

/// 'llvm::ImmutableMap' traits for maps from a variable name to 'Value'.
template <typename Value>
struct VariableMapInfo : llvm::ImutKeyValueInfo<std::string, Value> {
  using Base = llvm::ImutKeyValueInfo<std::string, Value>;

  using key_type_ref = llvm::StringRef;

  // Members needed to make read-only methods work with 'llvm::StringRef' rather
  // than 'std::string'.

  static key_type_ref KeyOfValue(typename Base::value_type_ref entry) {
    return entry.first;
  }

  static bool isEqual(key_type_ref lhs, key_type_ref rhs) { return lhs == rhs; }

  static bool isLess(key_type_ref lhs, key_type_ref rhs) { return lhs < rhs; }

  static void Profile(llvm::FoldingSetNodeID &id,
                      typename Base::value_type_ref entry) {
    id.AddString(entry.first);
    llvm::ImutProfileInfo<Value>::Profile(id, entry.second);
  }
};

/// Map from the names of the variables a variable tracking type system tracks
/// to the 'Value' each of them is tracked with.
///
/// The map is backed by an immutable (persistent) data structure since typing
/// contexts are copied frequently during generation: copying it is thus cheap,
/// while tracking or untracking one more variable only allocates the nodes
/// needed to represent the change, structurally sharing everything else with
/// the map it was derived from.
template <typename Value>
class VariableMap {
  using Map = llvm::ImmutableMap<std::string, Value, VariableMapInfo<Value>>;

public:
  using Factory = typename Map::Factory;

  /// Constructs an empty map.
  VariableMap() = default;

  /// Returns the value 'name' is tracked with or null if 'name' is not tracked.
  const Value *lookup(llvm::StringRef name) const { return map.lookup(name); }

  /// Returns whether 'name' is tracked.
  bool contains(llvm::StringRef name) const { return map.contains(name); }

  /// Returns this map with 'name' mapped to 'value', replacing the value
  /// 'name' is currently tracked with, if any.
  [[nodiscard]] VariableMap insert(Factory &factory, llvm::StringRef name,
                                   const Value &value) const {
    return VariableMap(factory.add(map, name, value));
  }

  /// Returns this map with 'name' no longer tracked.
  [[nodiscard]] VariableMap erase(Factory &factory,
                                  llvm::StringRef name) const {
    return VariableMap(factory.remove(map, name));
  }

  auto begin() const { return map.begin(); }
  auto end() const { return map.end(); }

private:
  explicit VariableMap(Map map) : map(std::move(map)) {}

  Map map{nullptr};
};

/// Value type for variable tracking type systems that only care about *which*
/// variables are tracked rather than about any data attached to them.
struct NoValue {
  bool operator==(NoValue) const { return true; }

  void Profile(llvm::FoldingSetNodeID &) const {}
};

/// Base class for the typing context of a 'VariableTrackingTypeSystemBase'.
///
/// Concrete type systems derive their own context from this and add whatever
/// state they need on top.
/// The 'VariableTrackingTypeSystemBase' below guarantees that 'variables' is
/// always the most up-to-date version of the map regardless of generation order
/// of sub-elements.
template <typename Value>
struct VariableTrackingTypingContext {
  /// Value the variables of this context are tracked with. Used by
  /// 'VariableTrackingTypeSystemBase' to deduce it from the context.
  using VariableValue = Value;

  /// The variables tracked so far.
  VariableMap<Value> variables;

  /// Returns the value 'name' is tracked with or null if 'name' is not tracked.
  const Value *lookupVariable(llvm::StringRef name) const {
    return variables.lookup(name);
  }

  /// Returns whether 'name' is tracked.
  bool isTracked(llvm::StringRef name) const {
    return variables.contains(name);
  }

  /// Merges 'rhs' into this context: the result is this context
  /// with the variable map of whichever of the two contexts was operated on
  /// more recently.
  ///
  /// Implementation detail of 'VariableTrackingTypeSystemBase'!
  template <typename Context>
  Context merge(const Context &rhs) const {
    static_assert(std::is_base_of_v<VariableTrackingTypingContext, Context>,
                  "'merge' must be called on the full derived context");

    Context result = static_cast<const Context &>(*this);
    if (rhs.clock > clock) {
      result.variables = rhs.variables;
      result.clock = rhs.clock;
    }
    return result;
  }

private:
  /// Number of track/untrack operations the variable map of this context has
  /// been built by. Since maps only ever change through those operations and
  /// each of them increments the clock, the map operated on most recently is
  /// the one with the greatest clock.
  /// This is used in 'merge' for 'VariableTrackingTypeSystemBase' to track the
  /// most recent map.
  std::uint64_t clock = 0;

  template <typename, typename>
  friend class VariableTrackingTypeSystemBase;
};

/// CRTP base class for type systems whose job is to track variables by name
/// through the program and map each of them to a value.
///
/// 'Context' must derive from 'VariableTrackingTypingContext', which is what
/// holds the tracked variables.
/// 'Self' is the class deriving from 'VariableTrackingTypeSystemBase'.
///
/// The class does not define any ordering in its transfer functions but does
/// guarantee that regardless of generation order, that every input context
/// always contains the most recent variables map.
/// Derived classes should use 'track' and 'untrack' whenever they want to add
/// or remove variables from the map.
///
/// These methods must be called from transfer functions which wrap the default
/// transfer functions (i.e. 'defaultTransferFn' and 'defaultOutputTransferFn').
/// The existing context passed to 'wrap' is guaranteed to already contain the
/// most recent variable map.
template <typename Context, typename Self>
class VariableTrackingTypeSystemBase : public CounterTypeSystem<Context, Self> {
public:
  /// Value the tracked variables are mapped to.
  using Value = typename Context::VariableValue;

  static_assert(
      std::is_base_of_v<VariableTrackingTypingContext<Value>, Context>,
      "'Context' must derive from 'VariableTrackingTypingContext'");

protected:
  /// Returns 'context' with 'name' tracked with 'value', replacing whatever
  /// state 'name' was in before.
  /// The most recent operation on a variable determines its state.
  [[nodiscard]] Context track(Context context, llvm::StringRef name,
                              const Value &value) const {
    context.variables = context.variables.insert(*factory, name, value);
    ++context.clock;
    return context;
  }

  /// Returns 'context' with 'name' no longer tracked.
  [[nodiscard]] Context untrack(Context context, llvm::StringRef name) const {
    context.variables = context.variables.erase(*factory, name);
    ++context.clock;
    return context;
  }

private:
  /// Factory backing every map produced for
  /// 'VariableTrackingTypingContext::variables'. Held behind a 'unique_ptr' as
  /// the type system itself must stay movable for conjunction, and as the maps
  /// in the contexts point to it.
  std::unique_ptr<typename VariableMap<Value>::Factory> factory =
      std::make_unique<typename VariableMap<Value>::Factory>();
};

} // namespace dynamatic::gen

#endif
