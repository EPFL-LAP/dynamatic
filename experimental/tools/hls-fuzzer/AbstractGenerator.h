#ifndef DYNAMATIC_HLS_FUZZER_ABSTRACTGENERATOR
#define DYNAMATIC_HLS_FUZZER_ABSTRACTGENERATOR

#include "Options.h"
#include "Randomly.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

namespace dynamatic {

/// Base class representing an instance of a generator + oracle.
/// The methods of this class are continuously used per worker thread to
/// discover bugs.
class AbstractGenerator {
public:
  explicit AbstractGenerator(const Options &options, Randomly &&random)
      : options(options), random(random) {}

  virtual ~AbstractGenerator();

  /// Creates a random C program that should be written to 'os'.
  /// 'os' writes to a file called 'functionName.c'.
  virtual void generate(llvm::raw_ostream &os,
                        llvm::StringRef functionName) const = 0;

  /// Result of verification.
  enum VerificationResult {
    /// A bug was found and a reproducer should be created.
    Bug,
    /// No bug was found and the next program can be generated.
    Success,
  };

  /// Verifies the property-under test
  /// (correctness or non-functional properties) of the given source file.
  /// The source file is guaranteed to have been previously written to by the
  /// 'generate' method.
  /// The directory the source file is contained in is empty and can/should be
  /// used as scratch space.
  ///
  /// If a bug is found, the entire directory is saved as a reproducer.
  virtual VerificationResult
  verify(const std::filesystem::path &sourceFile) const = 0;

protected:
  const Options &options;
  mutable Randomly random;
};

} // namespace dynamatic

#endif
