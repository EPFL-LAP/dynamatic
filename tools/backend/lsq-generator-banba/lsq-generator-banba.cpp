#include "BanbaDialect.h"
#include "LsqConfig.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::OptionCategory banbaCategory("lsq-generator-banba options");

static llvm::cl::opt<std::string>
    configFile("config", llvm::cl::desc("Path to the LSQ's JSON config file"),
               llvm::cl::Required, llvm::cl::cat(banbaCategory));

int main(int argc, char **argv) {
  // Only show this program's own options in --help
  llvm::cl::HideUnrelatedOptions(banbaCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Loading a CIRCT dialect makes this program depend on CIRCT's headers and
  // libraries, which checks that Dynamatic's build finds both
  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  context.loadDialect<banba::BanbaDialect>();

  // Read the config file
  auto buffer = llvm::MemoryBuffer::getFile(configFile);
  if (!buffer) {
    llvm::errs() << "Cannot read " << configFile << ": "
                 << buffer.getError().message() << "\n";
    return 1;
  }

  // Parse its content as JSON, into the struct
  llvm::Expected<LsqConfig> config =
      llvm::json::parse<LsqConfig>((*buffer)->getBuffer(), "LSQ config");
  if (!config) {
    llvm::errs() << "Cannot parse " << configFile << ": "
                 << llvm::toString(config.takeError()) << "\n";
    return 1;
  }

  // Stop if a switch of the pre-existing LSQ generators
  // has a value other than the one Banba requires
  if (!hasRequiredValues(*config))
    return 1;

  // Store the LSQ's parameters in the Banba dialect's attribute, and print it
  banba::ConfigAttr configAttr = banba::ConfigAttr::get(&context, *config);
  llvm::outs() << configAttr << "\n";
  return 0;
}
