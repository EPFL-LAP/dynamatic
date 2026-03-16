#ifndef DYNAMATIC_HLS_FUZZER_OPTIONSPARSER
#define DYNAMATIC_HLS_FUZZER_OPTIONSPARSER

#include "Options.h"

#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/StringSaver.h>

namespace dynamatic {

class OptionsParser : llvm::opt::GenericOptTable {
public:
  OptionsParser(llvm::ArrayRef<char *> args);

  /// Returns true if '--help' was specified.
  bool shouldDisplayHelp() const;

  void printHelp(llvm::raw_ostream &os) const {
    GenericOptTable::printHelp(os, "hls-fuzzer <dynamatic-path>", "hls-fuzzer");
  }

  /// Returns the number of generator threads that should be used for fuzzing.
  std::optional<std::size_t> getNumThreads() const;

  /// Returns the name of the target fuzzer.
  std::string getTargetName() const;

  /// Returns the positional arguments.
  std::vector<std::string> getPositionalArguments() const;

  /// Applies all commandline options to the options struct.
  Options apply(Options defaults);

private:
  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver stringSaver;
  llvm::opt::InputArgList args;
};

} // namespace dynamatic
#endif
