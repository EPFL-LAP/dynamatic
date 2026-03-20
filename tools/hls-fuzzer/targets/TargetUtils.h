#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_TARGETUTILS
#define DYNAMATIC_HLS_FUZZER_TARGETS_TARGETUTILS

#include "hls-fuzzer/AbstractWorker.h"

namespace dynamatic {

/// Performs differential testing of a C file called 'sourceFile'.
/// The C file needs to adhere to the usual integration test workflow used by
/// dynamatic, i.e., use 'CALL_KERNEL' and pass variables as arguments that are
/// identical to the corresponding parameter names.
/// 'dynamaticPath' should refer to where the dynamatic executable.
///
/// The directory of 'sourceFile' is assumed to be scratch space used for build
/// artifacts.
AbstractWorker::VerificationResult
performDifferentialTesting(const std::filesystem::path &sourceFile,
                           llvm::StringRef dynamaticPath);

} // namespace dynamatic

#endif
