//===- dynamatic.cpp - Dynamatic frontend -----------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool implements a (barebone, at this point) shell/frontend for
// Dynamatic, allowing users to go from C to VHDL using a simple command syntax.
// See the sample scripts in samples/ to get an idea of the syntax, or type
// 'help' in the shell to see a list of available commands and theit syntax. The
// sample scripts can be executed automatically on shell startup with the
// following command (from Dynamatic's top-level directory):
//
// ```sh
// ./bin/dynamatic --run=tools/dynamatic/samples/<script-name>.sh
// ```
//
// The tool severely lacks documentation (and cleanliness) at this point. This
// will all be fixed in future releases.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/System.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <readline/history.h>
#include <readline/readline.h>
#include <sstream>

using namespace llvm;
using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;

static constexpr llvm::StringLiteral ERR("[ERROR] "),
    DELIM("============================================="
          "===================================\n"),
    PROMPT("dynamatic> "), CMD_SET_SRC("set-src");

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string>
    run("run", cl::Optional,
        cl::desc("Path to a text file containing a sequence of commands to run "
                 "on startup."),
        cl::init(""), cl::cat(mainCategory));

static cl::opt<bool> exitOnFailure(
    "exit-on-failure", cl::Optional,
    cl::desc(
        "If specified, exits the frontend automatically on command failure"),
    cl::init(false), cl::cat(mainCategory));

namespace {
enum class CommandResult { SYNTAX_ERROR, FAIL, SUCCESS, EXIT, HELP };
} // namespace

template <typename... Tokens>
static CommandResult execCmd(Tokens... tokens) {
  return exec({tokens...}) != 0 ? CommandResult::FAIL : CommandResult::SUCCESS;
}

std::string floatToString(double f, size_t nDecimalPlaces) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(nDecimalPlaces) << f;
  return ss.str();
}

namespace {

struct FrontendState {
  std::string cwd;
  std::string dynamaticPath;
  std::string polygeistPath;
  // By default, the clock period is 4 ns
  double targetCP = 4.0;
  std::optional<std::string> sourcePath = std::nullopt;

  FrontendState(StringRef cwd) : cwd(cwd), dynamaticPath(cwd){};

  bool sourcePathIsSet(StringRef keyword);

  std::string getScriptsPath() const {
    return dynamaticPath + "/tools/dynamatic/scripts";
  }

  inline std::string getSeparator() const {
    return sys::path::get_separator().str();
  }

  inline std::string getKernelDir() const {
    assert(sourcePath && "source path not set");
    return path::parent_path(*sourcePath).str();
  }

  inline std::string getKernelName() const {
    assert(sourcePath && "source path not set");
    return path::filename(*sourcePath).drop_back(2).str();
  }

  inline std::string getOutputDir() const {
    return getKernelDir() + getSeparator() + "out";
  }

  std::string makeAbsolutePath(StringRef path);
};

struct Argument {
  StringRef name;
  StringRef desc;

  Argument() = default;

  Argument(StringRef name, StringRef desc) : name(name), desc(desc){};
};

struct CommandArguments {
  SmallVector<StringRef> positionals;
  mlir::DenseSet<StringRef> flags;
  StringMap<StringRef> options;
};

class Command {
public:
  StringRef keyword;
  StringRef desc;

  StringMap<Argument> positionals;
  StringMap<Argument> flags;
  StringMap<Argument> options;

  Command(StringRef keyword, StringRef desc, FrontendState &state)
      : keyword(keyword), desc(desc), state(state) {}

  void addPositionalArg(const Argument &arg) {
    assert(!positionals.contains(arg.name) && "duplicate positional arg name");
    positionals[arg.name] = arg;
  }

  void addFlag(const Argument &arg) {
    assert(!flags.contains(arg.name) && "duplicate flag name");
    assert(!options.contains(arg.name) && "option and flag have same name");
    flags[arg.name] = arg;
  }

  void addOption(const Argument &arg) {
    assert(!options.contains(arg.name) && "duplicate option name");
    assert(!flags.contains(arg.name) && "option and flag have same name");
    options[arg.name] = arg;
  }

  CommandResult parseAndExecute(ArrayRef<std::string> tokens);

  virtual CommandResult execute(CommandArguments &args) = 0;

  std::string getShortCmdDesc() const;

  void help() const;

  virtual ~Command() = default;

protected:
  FrontendState &state;

  inline std::string getSeparator() const { return state.getSeparator(); }

private:
  LogicalResult parsePositional(StringRef arg, CommandArguments &args) const;

  LogicalResult parseFlag(StringRef name, CommandArguments &args) const;

  LogicalResult parseOption(StringRef name, StringRef value,
                            CommandArguments &args) const;
};

class Exit : public Command {
public:
  Exit(FrontendState &state)
      : Command("exit", "Exits the Dynamatic frontend", state){};

  CommandResult execute(CommandArguments &args) override;
};

class Help : public Command {
public:
  Help(FrontendState &state)
      : Command("help", "Displays this help message", state){};

  CommandResult execute(CommandArguments &args) override;
};

class SetDynamaticPath : public Command {
public:
  SetDynamaticPath(FrontendState &state)
      : Command("set-dynamatic-path",
                "Sets the path to Dynamatic's top-level directory", state) {
    addPositionalArg({"path", "path to Dynamatic's top-level directory"});
  }

  CommandResult execute(CommandArguments &args) override;
};

class SetPolygeistPath : public Command {
public:
  SetPolygeistPath(FrontendState &state)
      : Command("set-polygeist-path",
                "Sets the path to Polygeist installation directory", state) {
    addPositionalArg({"path", "path to Polygeist installation directory"});
  }

  CommandResult execute(CommandArguments &args) override;
};

class SetSrc : public Command {
public:
  SetSrc(FrontendState &state)
      : Command(CMD_SET_SRC, "Sets the C source to compile", state) {
    addPositionalArg({"source", "path to source file"});
  }

  CommandResult execute(CommandArguments &args) override;
};

class SetCP : public Command {
public:
  SetCP(FrontendState &state)
      : Command("set-clock-period", "Sets the clock period", state) {
    addPositionalArg({"clock-period", "clock period in ns"});
  }
  CommandResult execute(CommandArguments &args) override;
};

class Compile : public Command {
public:
  static constexpr llvm::StringLiteral BUFFER_ALGORITHM = "buffer-algorithm";
  static constexpr llvm::StringLiteral SHARING = "sharing";

  Compile(FrontendState &state)
      : Command("compile",
                "Compiles the source kernel into a dataflow circuit; "
                "produces both handshake-level IR and an equivalent DOT file",
                state) {
    addOption(
        {BUFFER_ALGORITHM,
         "The buffer placement algorithm to use, values are "
         "'on-merges' (default option: minimum buffering for "
         "correctness), 'fpga20' (throughput-driven buffering), "
         "'fpl22' (throughput- and timing-driven buffering)"
         "'mapbuf' (simultaneous technology mapping and buffer placement)"});
    addFlag({SHARING, "Use credit-based resource sharing"});
  }

  CommandResult execute(CommandArguments &args) override;
};

class WriteHDL : public Command {
public:
  static constexpr llvm::StringLiteral HDL = "hdl";

  WriteHDL(FrontendState &state)
      : Command(
            "write-hdl",
            "Converts the DOT file produced after compile to VHDL using the "
            "export-dot tool",
            state) {
    addOption({HDL, "HDL to use for design's top-level"});
  }

  CommandResult execute(CommandArguments &args) override;
};

class Simulate : public Command {
public:
  Simulate(FrontendState &state)
      : Command("simulate",
                "Simulates the VHDL produced during HDL writing using Modelsim "
                "and the hls-verifier tool",
                state) {}

  CommandResult execute(CommandArguments &args) override;
};

class Visualize : public Command {
public:
  Visualize(FrontendState &state)
      : Command(
            "visualize",
            "Visualizes the execution of the circuit simulated by Modelsim.",
            state) {}

  CommandResult execute(CommandArguments &args) override;
};

class Synthesize : public Command {
public:
  Synthesize(FrontendState &state)
      : Command("synthesize",
                "Synthesizes the VHDL produced during HDL writing using Vivado",
                state) {}

  CommandResult execute(CommandArguments &args) override;
};

class FrontendCommands {
public:
  StringMap<std::unique_ptr<Command>> cmds;

  FrontendCommands() = default;

  template <typename Cmd>
  void add(FrontendState &state) {
    std::unique_ptr<Cmd> newCmd = std::make_unique<Cmd>(state);
    if (cmds.contains(newCmd->keyword)) {
      llvm::errs() << "Multiple commands exist with keyword '"
                   << newCmd->keyword << "'\n.";
      exit(1);
    }
    cmds[newCmd->keyword.str()] = std::move(newCmd);
  }

  bool contains(StringRef keyword) { return cmds.contains(keyword); }

  Command &get(StringRef keyword) {
    assert(cmds.contains(keyword));
    return *cmds[keyword];
  }
};
} // namespace

std::string FrontendState::makeAbsolutePath(StringRef path) {
  SmallString<128> str;
  path::append(str, path);
  fs::make_absolute(cwd, str);
  return str.str().str();
}

bool FrontendState::sourcePathIsSet(StringRef keyword) {
  if (!sourcePath.has_value()) {
    llvm::errs() << ERR
                 << "The path to the source file needs to be set to run '"
                 << keyword << "' use the '" << CMD_SET_SRC
                 << "' command before '" << keyword << "'.\n";
    return false;
  }
  return true;
}

CommandResult Command::parseAndExecute(ArrayRef<std::string> tokens) {
  // Don't report an error if the command is just empty
  if (tokens.empty())
    return CommandResult::SUCCESS;

  CommandArguments parsed;
  ArrayRef<std::string> opts = tokens.drop_front();
  for (const auto *tokIt = opts.begin(); tokIt != opts.end(); ++tokIt) {
    StringRef tok = *tokIt;
    if (tok.starts_with("--")) {
      // Flag or option
      StringRef name = tok.drop_front(2);
      if (flags.contains(name)) {
        // This is a flag
        if (failed(parseFlag(name, parsed)))
          return CommandResult::SYNTAX_ERROR;
      } else if (options.contains(name)) {
        // This is an option
        const auto *nextToken = ++tokIt;
        if (nextToken == opts.end()) {
          llvm::errs() << "Missing value for option '" << tok << "'\n";
          return CommandResult::SYNTAX_ERROR;
        }
        if (failed(parseOption(name, *nextToken, parsed)))
          return CommandResult::SYNTAX_ERROR;
      } else {
        llvm::errs() << ERR << "Unknow flag/option '" << tok << "'\n";
        return CommandResult::SYNTAX_ERROR;
      }
    } else if (failed(parsePositional(tok, parsed))) {
      // Positional argument
      return CommandResult::SYNTAX_ERROR;
    }
  }

  return execute(parsed);
}

LogicalResult Command::parsePositional(StringRef arg,
                                       CommandArguments &args) const {
  // Positional argument
  if (args.positionals.size() == positionals.size()) {
    llvm::outs() << ERR << "Expected only " << positionals.size()
                 << " argument for " << keyword << " command, but got extra '"
                 << arg << "'.\n";
    return failure();
  }
  args.positionals.push_back(arg);
  return success();
};

LogicalResult Command::parseFlag(StringRef name, CommandArguments &args) const {
  if (args.flags.contains(name)) {
    llvm::errs() << ERR << "Flag '" << name << "' given more than once\n";
    return failure();
  }
  args.flags.insert(name);
  return success();
};

LogicalResult Command::parseOption(StringRef name, StringRef value,
                                   CommandArguments &args) const {
  if (args.options.contains(name)) {
    llvm::errs() << ERR << "Option '" << name << "' given more than once\n";
    return failure();
  }
  args.options.insert({name, value});
  return success();
};

std::string Command::getShortCmdDesc() const {
  std::stringstream ss;
  ss << keyword.str() << " ";
  if (!flags.empty())
    ss << "[options] ";
  for (auto &nameAndArg : positionals)
    ss << "<" << nameAndArg.first().str() << "> ";
  return ss.str();
}

void Command::help() const {
  mlir::raw_indented_ostream os(llvm::outs());
  os << "USAGE: " << getShortCmdDesc() << "\n\n";

  auto printListArgs =
      [&](const StringMap<Argument> &args, const std::string &catName,
          const std::function<void(StringRef)> &fmtArg) -> void {
    if (args.empty())
      return;
    os << catName << ":\n";
    size_t maxLength = 0;
    std::vector<StringRef> posArgsStr;
    for (auto &nameAndArg : args)
      maxLength = std::max(maxLength, nameAndArg.second.name.size());

    os.indent();
    for (auto &nameAndArg : args) {
      const Argument &arg = nameAndArg.second;
      fmtArg(arg.name);
      os << std::string(maxLength - arg.name.size(), ' ') << " - " << arg.desc
         << "\n";
    }
    os.unindent();
    os << "\n";
  };

  printListArgs(positionals, "ARGUMENTS",
                [&](auto ref) { os << "<" << ref << ">"; });
  printListArgs(flags, "FLAGS", [&](auto ref) { os << "--" << ref; });
  printListArgs(options, "OPTIONS",
                [&](auto ref) { os << "--" << ref << " <option-value>"; });
  os << "\n";
}

CommandResult Exit::execute(CommandArguments &args) {
  return CommandResult::EXIT;
}

CommandResult Help::execute(CommandArguments &args) {
  return CommandResult::HELP;
}

CommandResult SetDynamaticPath::execute(CommandArguments &args) {
  // Remove the separator at the end of the path if there is one
  StringRef sep = sys::path::get_separator();
  std::string dynamaticPath = args.positionals.front().str();
  if (StringRef(dynamaticPath).ends_with(sep))
    dynamaticPath = dynamaticPath.substr(0, dynamaticPath.size() - 1);

  // Check whether the path makes sense
  if (!fs::exists(dynamaticPath + sep + "polygeist")) {
    llvm::outs() << ERR << "'" << dynamaticPath
                 << "' doesn't seem to point to Dynamatic, expected to "
                    "find, for example, a directory named 'polygeist' there.\n";
    return CommandResult::FAIL;
  }
  if (!fs::exists(dynamaticPath + sep + "bin")) {
    llvm::outs() << ERR
                 << "No 'bin' directory in provided path, Dynamatic doesn't "
                    "seem to have been built.\n";
    return CommandResult::FAIL;
  }

  state.dynamaticPath = state.makeAbsolutePath(dynamaticPath);
  return CommandResult::SUCCESS;
}

CommandResult SetPolygeistPath::execute(CommandArguments &args) {
  // Remove the separator at the end of the path if there is one
  StringRef sep = sys::path::get_separator();
  std::string polygeistPath = args.positionals.front().str();
  if (StringRef(polygeistPath).ends_with(sep))
    polygeistPath = polygeistPath.substr(0, polygeistPath.size() - 1);

  // Check whether the path makes sense
  if (!fs::exists(polygeistPath + sep + "llvm-project/")) {
    llvm::outs()
        << ERR << "'" << polygeistPath
        << "' doesn't seem to point to Polygeist, expected to "
           "find, for example, a directory named 'llvm-project/' there.\n";
    return CommandResult::FAIL;
  }
  if (!fs::exists(polygeistPath + sep + "build/bin/")) {
    llvm::outs() << ERR
                 << "No 'bin' directory in provided path, Polygeist doesn't "
                    "seem to have been built.\n";
    return CommandResult::FAIL;
  }

  state.polygeistPath = state.makeAbsolutePath(polygeistPath);
  return CommandResult::SUCCESS;
}

CommandResult SetSrc::execute(CommandArguments &args) {
  std::string sourcePath = args.positionals.front().str();
  StringRef srcName = path::filename(sourcePath);
  if (!srcName.ends_with(".c")) {
    llvm::outs() << ERR
                 << "Expected source file to have .c extension, but got '"
                 << path::extension(srcName) << "'.\n";
    return CommandResult::FAIL;
  }

  state.sourcePath = state.makeAbsolutePath(sourcePath);
  return CommandResult::SUCCESS;
}

CommandResult SetCP::execute(CommandArguments &args) {
  // Parse the float argument and check if the argument is legal.
  if (llvm::to_float(args.positionals.front().str(), state.targetCP))
    return CommandResult::SUCCESS;
  llvm::outs() << ERR << "Specified CP = " << args.positionals.front().str()
               << " is illegal.\n";
  return CommandResult::FAIL;
}

CommandResult Compile::execute(CommandArguments &args) {
  // We need the source path to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  std::string script = state.getScriptsPath() + getSeparator() + "compile.sh";
  // If unspecified, we place a OB + TB after every merge to guarantee
  // the deadlock freeness.
  std::string buffers = "on-merges";

  if (auto it = args.options.find(BUFFER_ALGORITHM); it != args.options.end()) {
    if (it->second == "on-merges" || it->second == "fpga20" ||
        it->second == "fpl22" || it->second == "mapbuf") {
      buffers = it->second;
    } else {
      llvm::errs()
          << "Unknown buffer placement algorithm " << it->second
          << "! Possible options are 'on-merges' (minimum buffering for "
             "correctness), 'fpga20' (throughput-driven buffering), or 'fpl22' "
             "(throughput- and timing-driven buffering). 'mapbuf' "
             "(simultaneous technology mapping and buffer placement)";
      return CommandResult::FAIL;
    }
  }

  std::string sharing = args.flags.contains(SHARING) ? "1" : "0";
  state.polygeistPath = state.polygeistPath.empty()
                            ? state.dynamaticPath + getSeparator() + "polygeist"
                            : state.polygeistPath;
  return execCmd(script, state.dynamaticPath, state.getKernelDir(),
                 state.getOutputDir(), state.getKernelName(), buffers,
                 floatToString(state.targetCP, 3), state.polygeistPath,
                 sharing);
}

CommandResult WriteHDL::execute(CommandArguments &args) {
  // We need the source path to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  std::string script = state.getScriptsPath() + getSeparator() + "write-hdl.sh";
  std::string hdl = "vhdl";

  if (auto it = args.options.find(HDL); it != args.options.end()) {
    if (it->second == "verilog") {
      hdl = "verilog";
    } else if (it->second != "vhdl") {
      llvm::errs() << "Unknow HDL '" << it->second
                   << "', possible options are 'vhdl' and "
                      "'verilog'.\n";
      return CommandResult::FAIL;
    }
  }

  return execCmd(script, state.dynamaticPath, state.getOutputDir(),
                 state.getKernelName(), hdl);
}

CommandResult Simulate::execute(CommandArguments &args) {
  // We need the source path to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  std::string script = state.getScriptsPath() + getSeparator() + "simulate.sh";
  return execCmd(script, state.dynamaticPath, state.getKernelDir(),
                 state.getOutputDir(), state.getKernelName());
}

CommandResult Visualize::execute(CommandArguments &args) {
  // We need the source path to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  std::string sep = getSeparator();
  std::string script = state.getScriptsPath() + sep + "visualize.sh";
  std::string dotPath = state.getOutputDir() + sep + "comp" + sep +
                        state.getKernelName() + ".dot";
  std::string wlfPath = state.getOutputDir() + sep + "sim" + sep +
                        "HLS_VERIFY" + sep + "vsim.wlf";

  return execCmd(script, state.dynamaticPath, dotPath, wlfPath,
                 state.getOutputDir(), state.getKernelName());
}

CommandResult Synthesize::execute(CommandArguments &args) {
  // We need the source path to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  std::string script =
      state.getScriptsPath() + getSeparator() + "synthesize.sh";

  return execCmd(script, state.dynamaticPath, state.getOutputDir(),
                 state.getKernelName(), floatToString(state.targetCP, 3),
                 floatToString(state.targetCP / 2, 3));
}

static StringRef removeComment(StringRef input) {
  if (size_t cutAt = input.find('#'); cutAt != std::string::npos)
    return input.take_front(cutAt);
  return input;
}

static void tokenizeInput(StringRef input, SmallVector<std::string> &tokens) {
  tokens.clear();
  std::istringstream inputStream(removeComment(input).str());
  std::string tok;
  while (inputStream >> tok)
    tokens.push_back(tok);
}

static void help(FrontendCommands &commands) {
  llvm::outs() << "List of available commands:\n\n";

  size_t maxLength = 0;
  std::vector<std::string> cmdFormats;
  for (auto &kwAndCmd : commands.cmds) {
    std::unique_ptr<Command> &cmd = kwAndCmd.second;
    std::string desc = cmd->getShortCmdDesc();
    maxLength = std::max(maxLength, desc.size());
    cmdFormats.push_back(desc);
  }

  mlir::raw_indented_ostream os(llvm::outs());
  os.indent();
  for (auto [fmt, kwAndCmd] : llvm::zip(cmdFormats, commands.cmds)) {
    std::unique_ptr<Command> &cmd = kwAndCmd.second;
    os << fmt << std::string(maxLength - fmt.size(), ' ') << " - " << cmd->desc
       << "\n";
  }
  os.unindent();
  llvm::outs() << "\n";
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Dynamatic Frontend");

  // Get current working directory
  SmallString<128> cwd;
  if (std::error_code ec = fs::current_path(cwd); ec.value() != 0) {
    llvm::errs() << "Failed to read current working directory.\n";
    return 1;
  }

  // Set up the frontend and available commands
  FrontendState state(cwd.str());
  FrontendCommands commands;
  commands.add<SetDynamaticPath>(state);
  commands.add<SetPolygeistPath>(state);
  commands.add<SetSrc>(state);
  commands.add<SetCP>(state);
  commands.add<Compile>(state);
  commands.add<WriteHDL>(state);
  commands.add<Simulate>(state);
  commands.add<Visualize>(state);
  commands.add<Synthesize>(state);
  commands.add<Help>(state);
  commands.add<Exit>(state);

  SmallVector<std::string> tokens;
  auto handleCmd = [&](StringRef input, bool prompt) -> void {
    tokenizeInput(input, tokens);
    if (tokens.empty())
      return;

    if (prompt)
      llvm::outs() << PROMPT << input << "\n";

    // Look for the command
    StringRef kw = tokens.front();
    if (!commands.contains(kw)) {
      llvm::outs() << ERR << "Unknown command '" << kw << "'.\n";
      help(commands);
      if (exitOnFailure)
        exit(1);
      return;
    }

    Command &cmd = commands.get(kw);

    // Decode the command that was identified via its keyword
    switch (cmd.parseAndExecute(tokens)) {
    case CommandResult::SYNTAX_ERROR:
      cmd.help();
      [[fallthrough]];
    case CommandResult::FAIL:
      if (!exitOnFailure)
        return;
      exit(1);
    case CommandResult::EXIT:
      llvm::outs() << "\nGoodbye!\n";
      exit(0);
    case CommandResult::HELP:
      help(commands);
      break;
    default:
      break;
    }
  };

  auto splitOnSemicolonAndHandle = [&](StringRef input, bool prompt) -> void {
    std::stringstream lineStream(removeComment(input).str());
    for (std::string cmd; std::getline(lineStream, cmd, ';');)
      handleCmd(cmd, prompt);
  };

  // Print frontend header
  llvm::outs()
      << DELIM +
             "============== Dynamatic | Dynamic High-Level Synthesis Compiler "
             "===============\n" +
             "======================== EPFL-LAP - v2.0.0 | March 2024 "
             "========================\n" +
             DELIM + "\n\n";

  // If a startup script is defined, we must run its commands first
  if (!run.empty()) {
    // Open the script
    std::ifstream inputFile(run);
    std::stringstream ss;
    if (!inputFile.is_open()) {
      llvm::errs() << "Failed to open startup script.\n";
      return 1;
    }

    // Read the script line-by-line and execute its commands
    // Supported delimeters: '\n' and ';'
    for (std::string scriptInput; std::getline(inputFile, scriptInput, '\n');)
      splitOnSemicolonAndHandle(scriptInput, true);
  }

  // Read from stdin, multiple commands in one line are separated by ';'
  // readline handles command history, allows user to repeat commands
  // with arrow keys
  while (char *rawInput = readline(PROMPT.str().c_str())) {
    add_history(rawInput);
    splitOnSemicolonAndHandle(std::string(rawInput), false);
    free(rawInput);
  }
  return 0;
}
