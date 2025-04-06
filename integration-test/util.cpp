#include "util.h"

#include <set>
#include <regex>

std::vector<fs::path> findTests(const fs::path &start) {
  std::set<std::string> ignored;
  fs::path ignorePath = start / "ignored_tests.txt";
  if (fs::exists(ignorePath)) {
    std::ifstream in(ignorePath);
    if (in.is_open()) {
      std::string line;
      while (getline(in, line)) {
        ignored.insert(line);
      }
    }
    else {
      std::cout << "[WARNING] Failed to open " << ignorePath << std::endl;
    }
  }

  std::vector<fs::path> ret;
  for (const auto &folder : fs::directory_iterator(start)) {
    if (folder.is_directory()) {
      for (const auto &entry : fs::directory_iterator(folder)) {
        if (ignored.find(entry.path().stem()) != ignored.end()) {
          std::cout << "[INFO] Ignoring " << entry.path() << std::endl;
        }
        else if (entry.is_regular_file() && entry.path().extension() == ".c") {
          ret.push_back(entry.path());
        }
      }
    }
  }

  return ret;
}

int runIntegrationTest(const fs::path &path, int &outSimTime) {
  std::string name = path.stem();
  std::cout << "[INFO] Running " << name << std::endl;
  std::string tmpFilename = "tmp_" + name + ".dyn";
  std::ofstream scriptFile(tmpFilename);
  if (!scriptFile.is_open()) {
    std::cout << "[ERROR] Failed to create .dyn script file" << std::endl;
    return -1;
  } 

  scriptFile << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
    << "set-src " << path.string() << std::endl
    << "compile" << std::endl
    << "write-hdl" << std::endl
    << "simulate" << std::endl
    << "exit" << std::endl;
  
  scriptFile.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  fs::path dynamaticLogPath = path.parent_path() / "out" / "dynamatic_out.txt";
  if (!fs::exists(dynamaticLogPath.parent_path())) {
    fs::create_directories(dynamaticLogPath);
  }

  std::string cmd = dynamaticPath.string() + " --exit-on-failure --run ";
  cmd += tmpFilename;
  cmd += " &> ";
  cmd += dynamaticLogPath;

  int status = system(cmd.c_str());
  if (status == 0) {
    fs::path logFilePath = path.parent_path() / "out" / "sim" / "report.txt";
    outSimTime = getSimulationTime(logFilePath);
  }

  return status;
}

int getSimulationTime(const fs::path &logFile) {
  std::ifstream file(logFile);
  if (!file.is_open()) {
    std::cout << "[WARNING] Failed to open " << logFile << std::endl;
    return -1;
  }

  std::vector<std::string> lines;
  std::string line;

  // Read all lines into a vector
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  std::regex pattern("Time: (\\d+) ns");
  std::smatch match;

  // Search lines in reverse order
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
      if (std::regex_search(*it, match, pattern)) {
          return std::stoi(match[1]) / 4;
      }
  }

  std::cout << "[WARNING] Log file does not contain simulation time!" << std::endl;
  return -1;
}