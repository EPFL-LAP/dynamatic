#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <llvm/ADT/StringSet.h>
#include <string>
#include <vector>

namespace dynamatic::experimental {

// TODO handle too many states to print
std::vector<std::string> getStateSet(const std::string &filename) {
  std::ifstream file(filename);
  std::vector<std::string> states;
  std::string line, currentState;
  bool recording = false;

  while (std::getline(file, line)) {

    // TODO put in function
    size_t start = line.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
      line.clear(); // The string contains only whitespace
    } else {
      // Trim leading and trailing spaces
      size_t end = line.find_last_not_of(" \t\n\r\f\v");
      line = line.substr(start, end - start + 1);
    }

    if (line.find("-------") != std::string::npos) {
      if (!currentState.empty()) {
        states.push_back(currentState);
      }
      currentState.clear();
      recording = true;
      continue;
    }

    if (!recording)
      continue;
    // Skip if it doesn't start with "miter."
    if (line.rfind("miter.", 0) != 0) {
      continue;
    }
    currentState += line + "\n";
  }

  if (!currentState.empty()) {
    states.push_back(currentState);
  }

  return states;
}

int getStates(const std::string &infFile, const std::string &finFile) {
  std::vector<std::string> finVector = getStateSet(finFile);
  std::vector<std::string> infVector = getStateSet(infFile);

  // TODO use StringSet directly
  llvm::StringSet<> setFin;
  llvm::StringSet<> setInf;
  for (const auto &entry : finVector) {
    setFin.insert(entry);
  }
  for (const auto &entry : infVector) {
    setInf.insert(entry);
  }

  int diffCount = 0;
  for (const auto &entry : infVector) {
    if (std::find(finVector.begin(), finVector.end(), entry) ==
        finVector.end()) {
      diffCount++;
    }
  }
  return diffCount;
}
} // namespace dynamatic::experimental
