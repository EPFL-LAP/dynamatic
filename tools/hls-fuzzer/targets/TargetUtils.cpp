#include "TargetUtils.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"

constexpr std::string_view EXECUTE_SCRIPT = "execute.sh";
constexpr std::string_view SHELL = "bash";

dynamatic::AbstractWorker::VerificationResult
dynamatic::performDifferentialTesting(const std::filesystem::path &sourceFile,
                                      llvm::StringRef dynamaticPath) {
  // Create an 'execute.sh' that can additionally be used as a nice reproducer
  // for e.g. 'cvise'.
  std::filesystem::path parentPath = sourceFile.parent_path();
  std::string executeFile = (parentPath / EXECUTE_SCRIPT).string();
  llvm::cantFail(llvm::writeToOutput(
      executeFile, [&](llvm::raw_ostream &os) -> llvm::Error {
        outputDynamaticInvocation(os, sourceFile, dynamaticPath, R"(
compile
write-hdl
simulate --timeout 20000
)");
        return llvm::Error::success();
      }));
  return executeInWorkingDirectory(parentPath,
                                   llvm::Twine(SHELL) + " " + EXECUTE_SCRIPT);
}

dynamatic::AbstractWorker::VerificationResult
dynamatic::performNonFunctionalTesting(
    const std::filesystem::path &sourceFile, llvm::StringRef dynamaticPath,
    llvm::StringRef oracleExecutable,
    llvm::ArrayRef<llvm::StringRef> arguments) {
  std::filesystem::path parentPath = sourceFile.parent_path();

  std::string executeFile = (parentPath / EXECUTE_SCRIPT).string();
  llvm::cantFail(llvm::writeToOutput(
      executeFile, [&](llvm::raw_ostream &os) -> llvm::Error {
        os << "set -e\n";
        outputDynamaticInvocation(os, sourceFile, dynamaticPath, R"(
compile
)");
        os << oracleExecutable;
        for (llvm::StringRef iter : arguments)
          os << " " << iter;

        os << '\n';
        return llvm::Error::success();
      }));
  return executeInWorkingDirectory(parentPath,
                                   llvm::Twine(SHELL) + " " + EXECUTE_SCRIPT);
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

  os << "set -o pipefail\n";
  os << "exec 5>&1\n";
  os << "OUTPUT=$(\n";

  // Perform constant evaluation with clang to find any instances of UB.
  os << R"(set -e
# Create a temporary file which has the 'static_assert' appended to it.
# Since we always append the 'static_assert' rather than making it part of the
# original source code, reduction tools such as 'cvise' cannot circumvent it.
file=$(mktemp --suffix .c)
trap 'rm "$file"' EXIT
)";
  os << "cat " << sourceFile.filename().string() << " >> $file\n";
  os << "echo \"static_assert((test_bench(), true));\"  >> $file\n";
  os << (dynamaticSourceRoot / "bin" / "clang++").string()
     << " $file -std=c++20 -DHLS_FUZZER_VERIFY ";
  os << "-I" << (dynamaticSourceRoot / "include").string()
     << " -Wno-deprecated -o /dev/null\n";

  // Invoke dynamatic.
  os << dynamaticPath << " --exit-on-failure <<EOF 2>&1 | tee >(cat - >&5)\n";
  os << "set-dynamatic-path " << dynamaticSourceRoot.string() << '\n';
  os << "set-src " << sourceFile.filename().string();
  os << "\n" << script.trim() << "\nexit\nEOF\n";
  os << R"()
RET=$?
# Ignore known issues.
if echo "$OUTPUT" | grep -q "Pointer values are unsupported"; then
  # See https://github.com/EPFL-LAP/dynamatic/issues/886
  exit 0
fi
# 'constexpr' evaluation does not define semantics for floating point edge
# cases that match implementation defined behaviour in clang, GCC and dynamatic.
if echo "$OUTPUT" | grep -q "floating point arithmetic produces a NaN"; then
  exit 0
fi
if [ "$RET" -ne "0" ]; then
  exit $RET;
fi
)";
}

dynamatic::AbstractWorker::VerificationResult
dynamatic::executeInWorkingDirectory(
    const std::filesystem::path &workingDirectory,
    const llvm::Twine &bashCommand) {

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
      "/usr/bin/bash", {SHELL, executeCWDFile}, /*Env=*/std::nullopt,
      /*Redirects=*/
      {"", (workingDirectory / "dynamatic-out.txt").string(),
       (workingDirectory / "dynamatic-err.txt").string()});
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
