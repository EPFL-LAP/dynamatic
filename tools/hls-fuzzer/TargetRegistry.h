#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_TARGETREGISTRY
#define DYNAMATIC_HLS_FUZZER_TARGETS_TARGETREGISTRY

#include "AbstractTarget.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace dynamatic {

/// Class acting as a registry for all targets registered.
class TargetRegistry {
public:
  TargetRegistry() = default;
  TargetRegistry(const TargetRegistry &) = delete;
  TargetRegistry &operator=(const TargetRegistry &) = delete;
  TargetRegistry(TargetRegistry &&) noexcept = default;
  TargetRegistry &operator=(TargetRegistry &&) noexcept = default;

  using Constructor = std::unique_ptr<AbstractTarget> (*)();

  /// Returns a range of 'StringRef's with the name of all targets.
  auto listTargets() const {
    return llvm::map_range(registry,
                           [](auto &&value) { return value.first(); });
  }

  /// Returns a new instance of the target of the given name or a null pointer
  /// if no such target exists.
  std::unique_ptr<AbstractTarget> getTarget(llvm::StringRef name) const;

  /// Registers a new target with the given name and constructor.
  void registerTarget(llvm::StringRef name, Constructor constructor) {
    registry.insert_or_assign(name, constructor);
  }

  /// Returns a singleton instance containing all globally registered targets.
  static TargetRegistry &getInstance() {
    static TargetRegistry instance;
    return instance;
  }

private:
  llvm::StringMap<Constructor> registry;
};

} // namespace dynamatic

/// Expands both arguments before pasting them together. Pasting '__LINE__'
/// directly would yield the literal token 'Register__LINE__', as '##'
/// suppresses the expansion of its operands.
#define DYNAMATIC_TARGET_CONCAT_(a, b) a##b
#define DYNAMATIC_TARGET_CONCAT(a, b) DYNAMATIC_TARGET_CONCAT_(a, b)

/// Registers a target in the global singleton registry with the given 'name'.
/// 'name' should be a string literal of the target name as it'll be used on
/// the command line. 'clazz' should be a subclass of 'AbstractTarget'.
#define REGISTER_TARGET(name, clazz)                                           \
  namespace {                                                                  \
  struct DYNAMATIC_TARGET_CONCAT(RegisterTarget, __LINE__) {                   \
    DYNAMATIC_TARGET_CONCAT(RegisterTarget, __LINE__)() {                      \
      ::dynamatic::TargetRegistry::getInstance().registerTarget(               \
          name, +[]() -> std::unique_ptr<dynamatic::AbstractTarget> {          \
            return std::make_unique<clazz>();                                  \
          });                                                                  \
    }                                                                          \
  } DYNAMATIC_TARGET_CONCAT(registerTarget, __LINE__);                         \
  }                                                                            \
  static_assert(true, "force semicolon")

#endif
