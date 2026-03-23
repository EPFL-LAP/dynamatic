#ifndef DYNAMATIC_HLS_FUZZER_OPTIONS
#define DYNAMATIC_HLS_FUZZER_OPTIONS

#include <string>

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
};

} // namespace dynamatic

#endif
