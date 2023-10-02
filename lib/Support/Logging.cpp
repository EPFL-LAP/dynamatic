//===- Logging.cpp - Utilities to create and manage log files ---*- C++ -*-===//
//
// Implementation of logging infrastructure.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/Logging.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm::sys;
using namespace dynamatic;

Logger::Logger(StringRef filepath, std::error_code &ec) {
  // Check whether the path contains a filename
  if (!path::has_filename(filepath)) {
    ec = std::error_code(errno, std::generic_category());
    return;
  }

  // Potentially create nested directories to create the logging file in
  logDir = path::parent_path(filepath);
  if (ec = fs::create_directories(logDir); ec.value() != 0)
    return;

  // Create the log file and store its file descriptor for writing
  if (logFile = new llvm::raw_fd_ostream(filepath, ec); ec.value() != 0)
    return;

  logStream = new mlir::raw_indented_ostream(*logFile);
}

Logger::~Logger() {
  // First delete the stream, the the file descriptor
  if (logStream)
    delete logStream;
  if (logFile)
    delete logFile;
}
