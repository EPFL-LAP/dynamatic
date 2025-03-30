#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>

int runIntegrationTest(const std::string& name);
std::vector<std::filesystem::path> findTests(const std::filesystem::path& start);

#endif // TEST_SUITE_H