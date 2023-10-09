//
// Created by Albert Fares on 05.10.2023.
//

#include "fileTools.h"
#include "structures.h"
#include <iostream>
#include <fstream>
#include <string>

FileResult openFile(const std::string& filename);
void closeFile(FileResult& result);