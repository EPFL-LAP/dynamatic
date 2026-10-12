#ifndef DYNAMATIC_HLS_FUZZER_REPRODUCER
#define DYNAMATIC_HLS_FUZZER_REPRODUCER

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dynamatic {

/// Name of the file dropped into every program's directory before generation
/// and read back by '--reproduce' to replay the exact same program.
inline constexpr llvm::StringLiteral REPRODUCER_FILE_NAME = "reproducer.json";

/// Everything needed to regenerate and re-verify a single program.
struct ReproducerInfo {
  /// Seed the program's randomness source was created with.
  std::uint32_t seed = 0;
  /// The target options the program was generated with, as command-line
  /// arguments that can be parsed again to reconstruct them. Storing the raw
  /// arguments rather than their parsed meaning keeps the reproducer in sync
  /// with the option definitions automatically.
  std::vector<std::string> arguments;
};

/// Writes 'info' to 'file' as JSON. Returns an error describing any failure to
/// do so rather than reporting it, leaving that decision to the caller.
llvm::Error writeReproducer(const std::filesystem::path &file,
                            const ReproducerInfo &info);

/// Reads a reproducer written by 'writeReproducer' back from 'file'. Returns
/// 'std::nullopt' if it could not be read, reporting the reason on the console.
std::optional<ReproducerInfo> readReproducer(const std::filesystem::path &file);

} // namespace dynamatic

#endif
