//
// Created by Albert Fares, Quentin Gross & Alice Potter on 05.10.2023.
//
#include "Parser.h"
#include <iostream>
#include <fstream>
#include "graphComponents.h"

class Parser {
public:
  explicit Parser(std::string filePath) : m_filePath(std::move(filePath)) {}

  bool parse(Graph* graph) {
    std::ifstream file(m_filePath);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << m_filePath << std::endl;
      return false;
    }

    std::string line;
    size_t lineIndex = 0;

    if (m_filePath.find(".csv") != std::string::npos){
      while (std::getline(file, line)) {
        processLine(line, graph, lineIndex);
        lineIndex++;
      }
    }
    else if (m_filePath.find(".dot") != std::string::npos){
      processAll(file, graph);
    }

    file.close();
    return true;
  }

private:
  std::string m_filePath;

protected:
  static void processLine(const std::string &line, Graph *graph, size_t lineIndex);
  static void processAll(std::ifstream &file, Graph *graph );
};