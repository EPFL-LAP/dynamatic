//===- Logging.h - Utilities to create and manage log files -----*- C++ -*-===//
//
// Utilities to facilitate interactions with log files by automatically managing
// underlying resources.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_LOGGING_H
#define DYNAMATIC_SUPPORT_LOGGING_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/Path.h"
#include <filesystem>

namespace dynamatic {

/// Manages a file descriptor and exposes a writer stream to it, both of which
/// are allocated on object creation and destroyed when the object goes out of
/// scope. Useful, for example, for logging debugging information to a file as
/// part of a pass.
class Logger {
public:
  /// Creates a logger at the given filepath. The path must include a filename
  /// thay may be preceded by a sequence of directories, which will be created
  /// if they do not exist. On error, `ec` will contain a non-zero error code
  /// and the logger should not be used.
  Logger(StringRef filepath, std::error_code &ec);

  /// Returns the path to the directory in which the log file is created.
  const std::string &getLogDir() { return logDir; }

  /// Returns a reference to an underlying indented writer stream to the log
  /// file.
  mlir::raw_indented_ostream &operator*() { return *logStream; }

  Logger(const Logger &other) = delete;
  Logger operator=(const Logger &other) = delete;

  /// Deallocates the indented writer stream and closes the log file's
  /// descriptor.
  ~Logger();

private:
  /// Output directory where the log file is created (without trailing
  /// separator).
  std::string logDir;
  /// Writer stream to the log file's descriptor.
  llvm::raw_fd_ostream *logFile = nullptr;
  /// Indented writer stream to the log file's descriptor.
  mlir::raw_indented_ostream *logStream = nullptr;
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_LOGGING_H