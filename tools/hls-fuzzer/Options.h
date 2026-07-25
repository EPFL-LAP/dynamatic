#ifndef DYNAMATIC_HLS_FUZZER_OPTIONS
#define DYNAMATIC_HLS_FUZZER_OPTIONS

#include <optional>
#include <string>
#include <vector>

namespace dynamatic {

enum class OracleKind {
  Functional,
  NonFunctional,
};

struct Options {
  // Path of this executable.
  std::string executablePath;
  std::string dynamaticExecutablePath;
  OracleKind kind = OracleKind::Functional;

  // Controls statistics reporting as requested via the '--statistics' option.
  // - If empty (nullopt), statistics collection is disabled.
  // - If present but the list is empty, all statistics are reported.
  // - Otherwise, only the named statistics are reported.
  std::optional<std::vector<std::string>> statistics;

  // File the fuzzing progress and statistics should be written to as JSON, as
  // requested via '--json-output'. If empty (nullopt), they are reported on the
  // console instead.
  std::optional<std::string> jsonOutput;
};

} // namespace dynamatic

#endif
