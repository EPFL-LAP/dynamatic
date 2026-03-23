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

/// Outputs a bash commandline to 'os' that invokes dynamatic and executes the
/// given 'script'.
/// 'sourceFile' is the source file to be compiled while 'dynamaticPath' refers
/// to the path to the dynamatic executable.
/// 'script' can assume that the source file and dynamatic home are already
/// set.
void outputDynamaticInvocation(llvm::raw_ostream &os,
                               const std::filesystem::path &sourceFile,
                               llvm::StringRef dynamaticPath,
                               llvm::StringRef script);

/// Executes a given 'bashCommand' in 'workingDirectory' by creating and
/// executing a script in 'workingDirectory'.
/// Normalizes the return code of the bash command to map failures to
/// 'VerificationResult::Bug'.
AbstractWorker::VerificationResult
executeInWorkingDirectory(const std::filesystem::path &workingDirectory,
                          llvm::StringRef bashCommand);

} // namespace dynamatic

#endif
