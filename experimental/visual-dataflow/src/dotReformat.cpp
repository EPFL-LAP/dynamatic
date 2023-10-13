//
// Created by Albert Fares on 02.10.2023.
//

#include <iostream>
#include <fstream>
#include <string>


void putPosOnSameLine(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);
    std::string line;
    std::string posBlock;


    bool isInPosBlock = false;

    while (std::getline(inputFile, line)) {
        if (line.find("pos=") != std::string::npos && line.find('e') != std::string::npos) {
            posBlock = line;
            isInPosBlock = true;
        } else if (isInPosBlock) {
            posBlock += ' ' + line;

            if (line.find("];") != std::string::npos) {
                isInPosBlock = false;
                outputFile << posBlock << '\n';

                outputFile << line << '\n';
            }
        } else {
            outputFile << line << '\n';
        }
    }

    inputFile.close();
    outputFile.close();

}


using namespace std;

void insertNewlineBeforeStyle(const string& inputFileName, const string& outputFileName) {
    ifstream inputFile(inputFileName);
    ofstream outputFile(outputFileName);


    string line;
    bool insertNewline = false;

    while (getline(inputFile, line)) {
        size_t posPos = line.find("pos=");
        size_t stylePos = line.find("style=");

        if (posPos != string::npos) {
            insertNewline = true;
        }

        if (insertNewline && stylePos != string::npos) {
            line.insert(stylePos, "\n");
            insertNewline = false;
        }

        outputFile << line << endl;
    }

    inputFile.close();
    outputFile.close();

}


void removeBackslashWithSpaceFromPos(const string& inputFileName, const string& outputFileName) {
    ifstream inputFile(inputFileName);
    ofstream outputFile(outputFileName);


    string line;

    while (getline(inputFile, line)) {
        size_t posPos = line.find("pos=");
        size_t backslashPos = line.find("\\ ");

        if (posPos != string::npos && backslashPos != string::npos && posPos < backslashPos) {
            string modifiedLine = line.substr(0, backslashPos) + line.substr(backslashPos + 2);
            outputFile << modifiedLine << endl;
        } else {
            outputFile << line << endl;
        }
    }

    inputFile.close();
    outputFile.close();

}


void removeEverythingAfterApostropheComma(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);
    std::string line;


    while (std::getline(inputFile, line)) {
        size_t posPos = line.find("pos=");
        size_t apostropheCommaPos = line.find("\",");

        if (posPos != std::string::npos && apostropheCommaPos != std::string::npos && posPos < apostropheCommaPos) {
            line = line.substr(0, apostropheCommaPos+2);
        }

        outputFile << line << '\n';
    }

    inputFile.close();
    outputFile.close();

}




void removeEverythingAfterCommaInStyle(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);
    std::string line;


    while (std::getline(inputFile, line)) {
        size_t stylePos = line.find("style=");
        size_t commaPos = line.find(',');

        if (stylePos != std::string::npos && commaPos != std::string::npos && stylePos < commaPos) {
            line = line.substr(0, commaPos);
        }

        outputFile << line << '\n';
    }

    inputFile.close();
    outputFile.close();

}

void reformatDot(const std::string& inputFileName, const std::string& outputFileName) {
    putPosOnSameLine(inputFileName, "rd1");
    insertNewlineBeforeStyle("rd1", "rd2");
    removeBackslashWithSpaceFromPos("rd2", "rd3");
    removeEverythingAfterCommaInStyle("rd3", "rd4");
    removeEverythingAfterApostropheComma("rd4",outputFileName);
    std::remove("rd1");
    std::remove("rd2");
    std::remove("rd3");
    std::remove("rd4");
}


