//
// Created by Albert Fares on 05.10.2023.
//

#include "fileTools.h"
#include "structures.h"
#include <iostream>
#include <fstream>
#include <string>



// Function to open a file and return a FileResult
FileResult openFile(const std::string& filename) {
  FileResult result;
  result.file.open(filename);

  if (!result.file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    result.status = FILE_NOT_FOUND;
  } else {
    result.status = NO_ERROR;
  }

  return result;
}

// Function to close the file
void closeFile(FileResult& result) {
  if (result.file.is_open()) {
    result.file.close();
  }
}

