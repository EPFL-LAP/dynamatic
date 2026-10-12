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
  // Arguments passed for options in the target options group, rendered so they
  // can be parsed again. Stored in the reproducer of every program so it can be
  // regenerated with the exact same options. See
  // 'OptionsParser::getTargetArguments'.
  std::vector<std::string> targetArguments;
  OracleKind kind = OracleKind::Functional;

  // Whether to skip verifying the generated program, as requested via the
  // internal '--no-verify' option. Used by tests to exercise generation and
  // reproduction without the heavy verification flow.
  bool noVerify = false;

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
