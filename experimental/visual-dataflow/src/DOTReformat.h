//===- DOTReformat.h - Reformats a .dot file -------------------*- C++ -*-===//
//
// The DOTReformat class is a temporary fix to the dot file being formatted wr-
// onfully so it can be parsed correctly.
//===----------------------------------------------------------------------===//

#include <string>

void putPosOnSameLine(const std::string &inputFileName,
                      const std::string &outputFileName);
void insertNewlineBeforeStyle(const std::string &inputFileName,
                              const std::string &outputFileName);
void removeBackslashWithSpaceFromPos(const std::string &inputFileName,
                                     const std::string &outputFileName);
void removeEverythingAfterApostropheComma(const std::string &inputFileName,
                                          const std::string &outputFileName);
void removeEverythingAfterCommaInStyle(const std::string &inputFileName,
                                       const std::string &outputFileName);
void reformatDot(const std::string &inputFileName,
                 const std::string &outputFileName);
