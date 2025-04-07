#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

int runIntegrationTest(const fs::path &name, int &outSimTime);
int getSimulationTime(const fs::path &logFile);
std::vector<fs::path> findTests(const fs::path &start);

#endif // UTIL_H