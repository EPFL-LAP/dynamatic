#include "OptionsParser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <thread>

namespace {
enum ID {
  OPT_INVALID = 0, // This is not a correct option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"

#undef OPTION
};

} // namespace

using namespace llvm::opt;

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr llvm::StringLiteral NAME##_init[] = VALUE;                  \
  static constexpr llvm::ArrayRef<llvm::StringLiteral> NAME(                   \
      NAME##_init, std::size(NAME##_init) - 1);
#include "Opts.inc"
#undef PREFIX

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"

#undef OPTION
};

dynamatic::OptionsParser::OptionsParser(llvm::ArrayRef<char *> args)
    : GenericOptTable(InfoTable), stringSaver(allocator),
      args(parseArgs(
          args.size(), args.data(), OPT_UNKNOWN, stringSaver,
          [](llvm::StringRef error) { llvm::report_fatal_error(error); })) {}

bool dynamatic::OptionsParser::shouldDisplayHelp() const {
  return args.hasArg(OPT_help);
}

std::optional<std::size_t> dynamatic::OptionsParser::getNumThreads() const {
  std::string maxThreads = std::to_string(std::thread::hardware_concurrency());
  std::size_t threads;
  if (args.getLastArgValue(OPT_num_threads, maxThreads)
          .getAsInteger(10, threads)) {
    llvm::errs() << "Expected integer instead of '"
                 << args.getLastArgValue(OPT_num_threads) << "'";
    return std::nullopt;
  }
  return threads;
}

std::string dynamatic::OptionsParser::getTargetName() const {
  return args.getLastArgValue(OPT_target).str();
}

std::vector<std::string>
dynamatic::OptionsParser::getPositionalArguments() const {
  return args.getAllArgValues(OPT_INPUT);
}

dynamatic::Options dynamatic::OptionsParser::apply(Options defaults) {
  std::vector<std::string> arguments = getPositionalArguments();
  if (arguments.size() == 1)
    defaults.dynamaticPath = getPositionalArguments()[0];

  return defaults;
}
