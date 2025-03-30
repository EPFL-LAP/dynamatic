#include "util.h"

std::vector<fs::path> findTests(const fs::path& start) {
  std::vector<fs::path> ret;
  for (const auto& folder : fs::directory_iterator(start)) {
    if (folder.is_directory()) {
      for (const auto& entry : fs::directory_iterator(folder)) {
          if (entry.is_regular_file() && entry.path().extension() == ".c") {
              ret.push_back(entry.path());
          }
      }
    }
  }

  return ret;
}

int runIntegrationTest(const fs::path& path) {
  std::string name = path.stem();
  std::cout << "Running " << name << std::endl;
  std::string tmpFilename = "tmp_" + name + ".dyn";
  std::ofstream script_file(tmpFilename);
  if (!script_file.is_open()) {
    std::cout << "Failed to create .dyn script file" << std::endl;
    return -1;
  } 

  script_file << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
    << "set-src " << path.string() << std::endl
    << "compile" << std::endl
    << "write-hdl" << std::endl
    << "simulate" << std::endl
    << "exit" << std::endl;
  
  script_file.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  std::string cmd = dynamaticPath.string() + " --exit-on-failure --run ";
  cmd += tmpFilename;
  return system(cmd.c_str());
}