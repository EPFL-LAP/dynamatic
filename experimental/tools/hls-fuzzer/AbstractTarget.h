#ifndef DYNAMATIC_HLS_FUZZER_ABSTRACTTARGET
#define DYNAMATIC_HLS_FUZZER_ABSTRACTTARGET

#include "AbstractGenerator.h"
#include "Options.h"
#include "Randomly.h"

#include <filesystem>
#include <memory>

namespace dynamatic {

/// Class representing a specific fuzzer target.
/// This is a customization point to add different fuzzers that may differ in
/// which oracle they use, or what kind of expressions they generate.
///
/// A target instance is selected using the '--target' command line option.
/// See 'TargetRegistry' for how to register a target.
class AbstractTarget {
public:
  virtual ~AbstractTarget() {}

  /// Creates a new generator with the given options and randomness source.
  /// This method is called for every worker thread.
  virtual std::unique_ptr<AbstractGenerator>
  createGenerator(const Options &options, Randomly randomly) const = 0;
};

} // namespace dynamatic

#endif
