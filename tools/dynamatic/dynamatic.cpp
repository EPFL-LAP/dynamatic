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
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace llvm;
using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;

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

const static std::string INFO = "[INFO] ";
const static std::string ERR = "[ERROR] ";
const static std::string DELIM = "============================================="
                                 "===================================\n";
const static std::string HEADER =
    DELIM +
    "============== Dynamatic | Dynamic High-Level Synthesis Compiler "
    "===============\n" +
    "======================= EPFL-LAP - v0.2.0 | November 2023 "
    "=======================\n" +
    DELIM + "\n\n";
const static std::string PROMPT = "dynamatic> ";

// Command names
const static std::string CMD_SET_SRC = "set-src";
const static std::string CMD_SET_DYNAMATIC_PATH = "set-dynamatic-path";
const static std::string CMD_SET_LEGACY_PATH = "set-legacy-path";
const static std::string CMD_COMPILE = "compile";
const static std::string CMD_WRITE_HDL = "write-hdl";
const static std::string CMD_SIMULATE = "simulate";
const static std::string CMD_SYNTHESIZE = "synthesize";
const static std::string CMD_HELP = "help";
const static std::string CMD_EXIT = "exit";

namespace {

struct FrontendState {
  std::string cwd;
  std::string dynamaticPath;
  std::optional<std::string> legacyPath = std::nullopt;
  std::optional<std::string> sourcePath = std::nullopt;

  FrontendState(StringRef cwd) : cwd(cwd), dynamaticPath(cwd){};

  bool sourcePathIsSet(StringRef keyword);

  bool legacyPathIsSet(StringRef keyword);

  std::string getScriptsPath() const {
    return dynamaticPath + "/tools/dynamatic/scripts";
  }

  std::string makeAbsolutePath(StringRef path);
};

struct Argument {
  StringRef name;
  StringRef desc;

  Argument() = default;

  Argument(StringRef name, StringRef desc) : name(name), desc(desc){};
};

enum class CommandResult { SYNTAX_ERROR, FAIL, SUCCESS, EXIT, HELP };

struct ParsedCommand {
  SmallVector<StringRef> positionals;
  mlir::DenseSet<StringRef> optArgsPresent;
};

class Command {
public:
  StringRef keyword;
  StringRef desc;
  StringMap<Argument> posArgs;
  StringMap<Argument> flags;

  Command(StringRef keyword, StringRef desc, FrontendState &state,
          SmallVector<Argument> &&posArgs = {},
          SmallVector<Argument> &&flags = {})
      : keyword(keyword), desc(desc), state(state) {
    for (Argument &arg : posArgs)
      this->posArgs[arg.name] = arg;
    for (Argument &arg : flags)
      this->flags[arg.name] = arg;
  };

  virtual CommandResult decode(SmallVector<std::string> &tokens) = 0;

  LogicalResult parse(SmallVector<std::string> &tokens, ParsedCommand &parsed);

  std::string getShortCmdDesc();

  void help();

  virtual ~Command() = default;

protected:
  FrontendState &state;
};

class Exit : public Command {
public:
  Exit(FrontendState &state)
      : Command(CMD_EXIT, "Exits the Dynamatic frontend", state){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class Help : public Command {
public:
  Help(FrontendState &state)
      : Command(CMD_HELP, "Displays this help message", state){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class SetDynamaticPath : public Command {
public:
  SetDynamaticPath(FrontendState &state)
      : Command(CMD_SET_DYNAMATIC_PATH,
                "Sets the path to Dynamatic's top-level directory", state,
                {{"path", "path to Dynamatic's top-level directory"}}){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class SetLegacyPath : public Command {
public:
  SetLegacyPath(FrontendState &state)
      : Command(CMD_SET_LEGACY_PATH,
                "Sets the path to legacy Dynamatic's top-level directory",
                state,
                {{"path", "path to legacy Dynamatic's top-level directory"}}){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class SetSrc : public Command {
public:
  SetSrc(FrontendState &state)
      : Command(CMD_SET_SRC, "Sets the C source to compile", state,
                {{"source", "path to source file"}}){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class Compile : public Command {
public:
  Compile(FrontendState &state)
      : Command(CMD_COMPILE,
                "Compiles the source kernel into a dataflow circuit; "
                "produces both handshake-level IR and an equivalent DOT file",
                state, {},
                {{"simple-buffers", "Use simple buffer placement"}}){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class WriteHDL : public Command {
public:
  WriteHDL(FrontendState &state)
      : Command(
            CMD_WRITE_HDL,
            "Converts the DOT file produced after compile to VHDL using the "
            "legacy dot2vhdl tool",
            state){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class Simulate : public Command {
public:
  Simulate(FrontendState &state)
      : Command(CMD_SIMULATE,
                "Simulates the VHDL produced during HDL writing using Modelsim "
                "and the legacy HLSVerifier tool",
                state){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
};

class Synthesize : public Command {
public:
  Synthesize(FrontendState &state)
      : Command(CMD_SYNTHESIZE,
                "Synthesizes the VHDL produced during HDL writing using Vivado",
                state){};

  CommandResult decode(SmallVector<std::string> &tokens) override;
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

static CommandResult execShellCommand(StringRef cmd) {
  int ret = std::system(cmd.str().c_str());
  llvm::outs() << "\n";
  return ret != 0 ? CommandResult::FAIL : CommandResult::SUCCESS;
}

std::string FrontendState::makeAbsolutePath(StringRef path) {
  SmallString<128> str;
  path::append(str, path);
  fs::make_absolute(cwd, str);
  return str.str().str();
}

bool FrontendState::sourcePathIsSet(StringRef keyword) {
  if (!sourcePath.has_value()) {
    llvm::outs() << ERR
                 << "The path to legacy Dynamatic needs to be set to run '"
                 << keyword << "' use the '" << CMD_SET_SRC
                 << "' command before '" << keyword << "'.\n";
    return false;
  }
  return true;
}

bool FrontendState::legacyPathIsSet(StringRef keyword) {
  if (!legacyPath.has_value()) {
    llvm::outs() << ERR << "The source needs to be set to run '" << keyword
                 << "' use the '" << CMD_SET_LEGACY_PATH << "' command before '"
                 << keyword << "'.\n";
    return false;
  }
  return true;
}

LogicalResult Command::parse(SmallVector<std::string> &tokens,
                             ParsedCommand &parsed) {
  bool firstIsKw = true;
  for (StringRef tok : tokens) {
    if (firstIsKw) {
      firstIsKw = false;
      continue;
    }
    if (tok.starts_with("--")) {
      StringRef flagName = tok.drop_front(2);
      if (!flags.contains(flagName)) {
        llvm::outs() << ERR << "Unknow flag '" << tok << "'\n";
        return failure();
      }
      if (parsed.optArgsPresent.contains(flagName)) {
        llvm::outs() << ERR << "Flag '" << tok
                     << "' indicated more than once\n";
        return failure();
      }
      parsed.optArgsPresent.insert(flagName);
    } else {
      if (parsed.positionals.size() == posArgs.size()) {
        llvm::outs() << ERR << "Expected only " << posArgs.size()
                     << " argument for " << keyword
                     << " command, but got extra '" << tok << "'.\n";
        return failure();
      }
      parsed.positionals.push_back(tok);
    }
  }
  return success();
}

std::string Command::getShortCmdDesc() {
  std::stringstream ss;
  ss << keyword.str() << " ";
  if (!flags.empty())
    ss << "[options] ";
  for (auto &nameAndArg : posArgs)
    ss << "<" << nameAndArg.first().str() << "> ";
  return ss.str();
}

void Command::help() {
  mlir::raw_indented_ostream os(llvm::outs());
  os << "USAGE: " << getShortCmdDesc() << "\n\n";

  auto printListArgs =
      [&](StringMap<Argument> &args, const std::string &catName,
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
      Argument &arg = nameAndArg.second;
      fmtArg(arg.name);
      os << std::string(maxLength - arg.name.size(), ' ') << " - " << arg.desc
         << "\n";
    }
    os.unindent();
    os << "\n";
  };

  printListArgs(posArgs, "ARGUMENTS",
                [&](auto ref) { os << "<" << ref << ">"; });
  printListArgs(flags, "OPTIONS", [&](auto ref) { os << "--" << ref; });
  os << "\n";
}

CommandResult Exit::decode(SmallVector<std::string> &tokens) {
  if (tokens.size() == 1)
    return CommandResult::EXIT;
  llvm::outs() << ERR << "To exit Dynamatic, just type 'exit'.\n";
  return CommandResult::FAIL;
}

CommandResult Help::decode(SmallVector<std::string> &tokens) {
  return CommandResult::HELP;
}

CommandResult SetDynamaticPath::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  // Add a slash at the end of the path if there isn't one already
  StringRef sep = sys::path::get_separator();
  std::string dynamaticPath = parsed.positionals.front().str();
  if (StringRef(dynamaticPath).ends_with(sep))
    dynamaticPath = dynamaticPath.substr(0, dynamaticPath.size() - 1);

  // Check whether the path makes sense
  if (!fs::exists(dynamaticPath + sep + "circt")) {
    llvm::outs() << ERR << "'" << dynamaticPath
                 << "' doesn't seem to point to Dynamatic, expected to "
                    "find, for example, a directory named 'circt' there.\n";
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

CommandResult SetLegacyPath::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  // Remove the eventual slash at the end of the path
  StringRef sep = sys::path::get_separator();
  std::string legacyPath = parsed.positionals.front().str();
  if (StringRef(legacyPath).ends_with(sep))
    legacyPath = legacyPath.substr(0, legacyPath.size() - 1);

  // Check whether the path makes sense
  if (!fs::exists(legacyPath + sep + "dot2vhdl")) {
    llvm::outs() << ERR << "'" << legacyPath
                 << "' doesn't seem to point to legacy Dynamatic, expected to "
                    "find, for example, a directory named 'dot2vhdl' there.\n";
    return CommandResult::FAIL;
  }

  state.legacyPath = state.makeAbsolutePath(legacyPath);
  return CommandResult::SUCCESS;
}

CommandResult SetSrc::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  std::string sourcePath = parsed.positionals.front().str();
  StringRef srcName = path::filename(sourcePath);
  if (!srcName.ends_with(".c")) {
    llvm::outs() << ERR
                 << "Expected source file to have .c extension, but got '"
                 << path::extension(srcName) << "'.\n";
    return CommandResult::FAIL;
  }

  if (!fs::exists(sourcePath)) {
    llvm::outs() << ERR << "File '" << sourcePath << "' does not exist.\n";
    return CommandResult::FAIL;
  }

  state.sourcePath = state.makeAbsolutePath(sourcePath);
  return CommandResult::SUCCESS;
}

CommandResult Compile::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  // We need the source and legacy paths to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  StringRef sep = sys::path::get_separator();
  std::string kernelDir = path::parent_path(*state.sourcePath).str();
  std::string kernelName = path::filename(*state.sourcePath).drop_back(2).str();
  std::string outputDir = kernelDir + sep.str() + "out";
  std::string buffers =
      parsed.optArgsPresent.contains("simple-buffers") ? "1" : "0";

  // Create and execute the command
  return execShellCommand(state.getScriptsPath() + "/compile.sh " +
                          state.dynamaticPath + " " + kernelDir + " " +
                          outputDir + " " + kernelName + " " + buffers);
}

CommandResult WriteHDL::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  // We need the source and legacy paths to be set
  if (!state.sourcePathIsSet(keyword) || !state.legacyPathIsSet(keyword))
    return CommandResult::FAIL;

  StringRef sep = sys::path::get_separator();
  std::string kernelDir = path::parent_path(*state.sourcePath).str();
  std::string kernelName = path::filename(*state.sourcePath).drop_back(2).str();
  std::string outputDir = kernelDir + sep.str() + "out";
  std::string dotPath = kernelDir + sep.str() + "out" + sep.str() + "comp" +
                        sep.str() + kernelName + ".dot";

  // The DOT file must exist to produce the corresponding VHDL
  if (!fs::exists(dotPath)) {
    llvm::outs() << ERR << "File '" << dotPath << "' does not exist.";
    return CommandResult::FAIL;
  }

  // Create and execute the command
  return execShellCommand(state.getScriptsPath() + "/write-hdl.sh " +
                          state.dynamaticPath + " " + *state.legacyPath + " " +
                          outputDir + " " + kernelName);
}

CommandResult Simulate::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  // We need the source and legacy paths to be set
  if (!state.sourcePathIsSet(keyword) || !state.legacyPathIsSet(keyword))
    return CommandResult::FAIL;

  StringRef sep = sys::path::get_separator();
  std::string kernelDir = path::parent_path(*state.sourcePath).str();
  std::string kernelName = path::filename(*state.sourcePath).drop_back(2).str();
  std::string outputDir = kernelDir + sep.str() + "out";
  std::string vhdlPath = kernelDir + sep.str() + "out" + sep.str() + "comp" +
                         sep.str() + kernelName + ".vhd";

  // The DOT file must exist to produce the corresponding VHDL
  if (!fs::exists(vhdlPath)) {
    llvm::outs() << ERR << "File '" << vhdlPath << "' does not exist.";
    return CommandResult::FAIL;
  }

  // Create and execute the command
  return execShellCommand(state.getScriptsPath() + "/simulate.sh " +
                          state.dynamaticPath + " " + *state.legacyPath + " " +
                          kernelDir + " " + outputDir + " " + kernelName);
}

CommandResult Synthesize::decode(SmallVector<std::string> &tokens) {
  ParsedCommand parsed;
  if (failed(parse(tokens, parsed)))
    return CommandResult::SYNTAX_ERROR;

  // We need the source and legacy paths to be set
  if (!state.sourcePathIsSet(keyword))
    return CommandResult::FAIL;

  StringRef sep = sys::path::get_separator();
  std::string kernelDir = path::parent_path(*state.sourcePath).str();
  std::string kernelName = path::filename(*state.sourcePath).drop_back(2).str();
  std::string outputDir = kernelDir + sep.str() + "out";
  std::string vhdlPath = kernelDir + sep.str() + "out" + sep.str() + "comp" +
                         sep.str() + kernelName + ".vhd";

  // The DOT file must exist to produce the corresponding VHDL
  if (!fs::exists(vhdlPath)) {
    llvm::outs() << ERR << "File '" << vhdlPath << "' does not exist.";
    return CommandResult::FAIL;
  }

  // Create and execute the command
  return execShellCommand(state.getScriptsPath() + "/synthesize.sh " +
                          state.dynamaticPath + " " + *state.legacyPath + " " +
                          outputDir + " " + kernelName);
}

static void tokenizeInput(StringRef input, SmallVector<std::string> &tokens) {
  tokens.clear();
  std::istringstream iss(input.str());
  std::string tok;
  while (iss >> tok) {
    if (StringRef(tok).starts_with("#"))
      return;
    tokens.push_back(tok);
  }
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

  // Set up the frontend end and available commands
  FrontendState state(cwd.str());
  FrontendCommands commands;
  commands.add<SetDynamaticPath>(state);
  commands.add<SetLegacyPath>(state);
  commands.add<SetSrc>(state);
  commands.add<Compile>(state);
  commands.add<WriteHDL>(state);
  commands.add<Simulate>(state);
  commands.add<Synthesize>(state);
  commands.add<Help>(state);
  commands.add<Exit>(state);

  auto handleInput = [&](StringRef input) -> void {
    SmallVector<std::string> tokens;
    tokenizeInput(input, tokens);
    if (tokens.empty())
      return;

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
    switch (cmd.decode(tokens)) {
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

  // Print frontend header
  llvm::outs() << HEADER;

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
    std::string line;
    while (std::getline(inputFile, line))
      if (!line.empty() && !StringRef(line).starts_with("#")) {
        llvm::outs() << PROMPT << line << "\n";
        handleInput(line);
      }
  }

  std::string userInput;
  while (true) {
    llvm::outs() << PROMPT;
    getline(std::cin, userInput);
    handleInput(userInput);
  }
  return 0;
}
