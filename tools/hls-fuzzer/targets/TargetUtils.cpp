#include "TargetUtils.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"

dynamatic::AbstractWorker::VerificationResult
dynamatic::performDifferentialTesting(const std::filesystem::path &sourceFile,
                                      llvm::StringRef dynamaticPath) {
  // Create an 'execute.sh' that can additionally be used as a nice reproducer
  // for e.g. 'cvise'.
  std::filesystem::path parentPath = sourceFile.parent_path();
  std::string executeFile = (parentPath / "execute.sh").string();
  llvm::cantFail(llvm::writeToOutput(
      executeFile, [&](llvm::raw_ostream &os) -> llvm::Error {
        os << dynamaticPath << " --exit-on-failure <<EOF\n";
        os << "set-dynamatic-path "
           << std::filesystem::path(dynamaticPath.str())
                  .parent_path()
                  .parent_path()
                  .string()
           << '\n';
        os << "set-src " << sourceFile.filename().string();
        os << R"(
compile
write-hdl
simulate
exit
EOF
)";
        return llvm::Error::success();
      }));

  // LLVM's process creation does not support changing the current working
  // directory. We require this since dynamatic creates many of its artifacts
  // in the working directory. Workaround this limitation using a wrapper
  // script that performs a 'cd' to the directory it is contained in.
  std::string executeCWDFile = (parentPath / "execute_cwd.sh").string();
  llvm::cantFail(llvm::writeToOutput(
      executeCWDFile, [&](llvm::raw_ostream &os) -> llvm::Error {
        os << R"a(SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR && bash )a"
           << executeFile
           // Canonicalize all error exists to exit code 1, even if dynamatic
           // crashed with e.g. SIGSEGV. We need this to differentiate between
           // bash exiting with a signal and dynamatic exiting with a signal.
           << "|| exit 1\n";
        return llvm::Error::success();
      }));

  int exitCode = llvm::sys::ExecuteAndWait(
      "/usr/bin/bash", {"bash", executeCWDFile}, /*Env=*/std::nullopt,
      /*Redirects=*/{"", "", ""});

  switch (exitCode) {
    // Normal exit.
  case 0:
    // bash (not dynamatic!) exited due to a signal. This is not a bug but the
    // user requesting our fuzzer (and its subprocesses) to exit via CTRL+C.
    // Count it as success rather than denoting it as a bug.
  case -2:
    return AbstractWorker::Success;
  default:
    return AbstractWorker::Bug;
  }
}
