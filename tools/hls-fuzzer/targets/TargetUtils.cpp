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
        outputDynamaticInvocation(os, sourceFile, dynamaticPath, R"(
compile
write-hdl
simulate
)");
        return llvm::Error::success();
      }));
  return executeInWorkingDirectory(parentPath, "bash execute.sh");
}

void dynamatic::outputDynamaticInvocation(
    llvm::raw_ostream &os, const std::filesystem::path &sourceFile,
    llvm::StringRef dynamaticPath, llvm::StringRef script) {
  // Compute the dynamatic home path by assuming it's a parent directory of the
  // dynamatic executable and contains the scripts directory used to implement
  // the various commands.
  std::filesystem::path dynamaticSourceRoot = dynamaticPath.str();
  while (!dynamaticSourceRoot.empty()) {
    dynamaticSourceRoot = dynamaticSourceRoot.parent_path();
    if (exists(dynamaticSourceRoot / "tools" / "dynamatic" / "scripts"))
      break;
  }

  os << dynamaticPath << " --exit-on-failure <<EOF\n";
  os << "set-dynamatic-path " << dynamaticSourceRoot.string() << '\n';
  os << "set-src " << sourceFile.filename().string();
  os << "\n" << script.trim() << "\nexit\nEOF\n";
}

dynamatic::AbstractWorker::VerificationResult
dynamatic::executeInWorkingDirectory(
    const std::filesystem::path &workingDirectory,
    llvm::StringRef bashCommand) {

  // LLVM's process creation does not support changing the current working
  // directory. We require this since dynamatic creates many of its artifacts
  // in the working directory. Workaround this limitation using a wrapper
  // script that performs a 'cd' to the directory it is contained in.
  std::string executeCWDFile = (workingDirectory / "execute_cwd.sh").string();
  llvm::cantFail(llvm::writeToOutput(
      executeCWDFile, [&](llvm::raw_ostream &os) -> llvm::Error {
        os << R"a(SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR && )a"
           << bashCommand
           // Canonicalize all error exists to exit code 1, even if dynamatic
           // crashed with e.g. SIGSEGV. We need this to differentiate between
           // bash exiting with a signal and dynamatic exiting with a signal.
           << " || exit 1\n";
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
