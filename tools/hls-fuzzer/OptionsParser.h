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

  /// Returns the number of programs that should be generated and verified
  /// before exiting, as requested via '--num-programs'. Returns
  /// 'std::nullopt' if the option was not specified, i.e. fuzzing should run
  /// until interrupted.
  std::optional<std::size_t> getNumPrograms() const;

  /// Returns the name of the target fuzzer.
  std::string getTargetName() const;

  /// Returns the statistics selection requested on the command line.
  /// Returns 'std::nullopt' if '--statistics' was not specified, an empty
  /// vector if it was specified without an explicit list (i.e. report all
  /// statistics), or the list of requested statistic names otherwise.
  std::optional<std::vector<std::string>> getStatistics() const;

  /// Returns the file that fuzzing progress and statistics should be written to
  /// as JSON, as requested via '--json-output'. Returns 'std::nullopt' if the
  /// option was not specified, i.e. they should be reported on the console.
  std::optional<std::string> getJSONOutput() const;

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
