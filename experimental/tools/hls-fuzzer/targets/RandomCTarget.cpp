#include "RandomCTarget.h"

#include "../BasicCProducer.h"
#include "../TargetRegistry.h"
#include "DynamaticTypeSystem.h"
#include "llvm/Support/Process.h"

REGISTER_TARGET("random-c", dynamatic::RandomCTarget);

using namespace dynamatic;

namespace {
class RandomCGenerator : public AbstractGenerator {
public:
  explicit RandomCGenerator(const Options &options, Randomly &&random)
      : AbstractGenerator(options, std::move(random)) {}

  void generate(llvm::raw_ostream &os,
                llvm::StringRef functionName) const override;

  VerificationResult
  verify(const std::filesystem::path &sourceFile) const override;
};

} // namespace

std::unique_ptr<dynamatic::AbstractGenerator>
dynamatic::RandomCTarget::createGenerator(const Options &options,
                                          Randomly randomly) const {
  return std::make_unique<RandomCGenerator>(options, std::move(randomly));
}

void RandomCGenerator::generate(llvm::raw_ostream &os,
                                llvm::StringRef functionName) const {
  gen::DynamaticTypeSystem dynamaticTypeSystem(random);
  gen::BasicCProducer generator(
      random, dynamaticTypeSystem,
      /*entryContext=*/
      {random.fromEnum<gen::DynamaticTypingContext::Constraint>()});

  ast::Function function = generator.generate(functionName);
  os << R"(
#include <stdint.h>
#include <math.h>
#include "dynamatic/Integration.h"

)";
  os << function << '\n';
  os << generator.generateTestBench(function);
}

AbstractGenerator::VerificationResult
RandomCGenerator::verify(const std::filesystem::path &sourceFile) const {

  // Create an 'execute.sh' that can additionally be used as a nice reproducer
  // for e.g. 'cvise'.
  std::filesystem::path parentPath = sourceFile.parent_path();
  std::string executeFile = (parentPath / "execute.sh").string();
  llvm::cantFail(llvm::writeToOutput(
      executeFile, [&](llvm::raw_ostream &os) -> llvm::Error {
        os << options.dynamaticPath << " --exit-on-failure <<EOF\n";
        os << "set-dynamatic-path "
           << std::filesystem::path(options.dynamaticPath)
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
           << executeFile << '\n';
        return llvm::Error::success();
      }));

  int exitCode = llvm::sys::ExecuteAndWait(
      "/usr/bin/bash", {"bash", executeCWDFile}, /*Env=*/std::nullopt,
      /*Redirects=*/{"", "", ""});
  return exitCode == 0 ? Success : Bug;
}
