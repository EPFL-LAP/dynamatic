#include <iostream>
#include <fstream>
#include <string>


void processDotFile(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);
    std::string line;
    std::string posBlock;


    bool isInPosBlock = false;

    while (std::getline(inputFile, line)) {
        if (line.find("pos=") != std::string::npos && line.find('e') != std::string::npos) {
            // Start of "pos=" block
            posBlock = line;
            isInPosBlock = true;
        } else if (isInPosBlock) {
            // Continue adding lines to "pos=" block
            posBlock += ' ' + line;

            if (line.find("];") != std::string::npos) {
                // End of "pos=" block
                isInPosBlock = false;
                outputFile << posBlock << '\n';

                // Add the current line as well (line after "pos=" block)
                outputFile << line << '\n';
            }
        } else {
            // Line doesn't contain "pos="; write it as is
            outputFile << line << '\n';
        }
    }

    inputFile.close();
    outputFile.close();

//    std::cout << "Processed " << inputFileName << " and saved as " << outputFileName << std::endl;
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

//    cout << "Modified file saved as " << outputFileName << endl;
}


void removeBackslashWithSpaceFromPos(const string& inputFileName, const string& outputFileName) {
    ifstream inputFile(inputFileName);
    ofstream outputFile(outputFileName);


    string line;

    while (getline(inputFile, line)) {
        size_t posPos = line.find("pos=");
        size_t backslashPos = line.find("\\ ");

        // Check if "pos" and " \\" are on the same line
        if (posPos != string::npos && backslashPos != string::npos && posPos < backslashPos) {
            // Remove the " \\" by copying the string without it
            string modifiedLine = line.substr(0, backslashPos) + line.substr(backslashPos + 2);
            outputFile << modifiedLine << endl;
        } else {
            outputFile << line << endl;
        }
    }

    inputFile.close();
    outputFile.close();

//    cout << "Modified file saved as " << outputFileName << endl;
}


void removeEverythingAfterApostropheComma(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);
    std::string line;


    while (std::getline(inputFile, line)) {
        size_t posPos = line.find("pos=");
        size_t apostropheCommaPos = line.find("\",");

        if (posPos != std::string::npos && apostropheCommaPos != std::string::npos && posPos < apostropheCommaPos) {
            // Truncate the line before the position of "\",\""
            line = line.substr(0, apostropheCommaPos+2);
        }

        outputFile << line << '\n';
    }

    inputFile.close();
    outputFile.close();

//    std::cout << "Processed " << inputFileName << " and saved as " << outputFileName << std::endl;
}




void removeEverythingAfterCommaInStyle(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);
    std::string line;


    while (std::getline(inputFile, line)) {
        size_t stylePos = line.find("style=");
        size_t commaPos = line.find(',');

        if (stylePos != std::string::npos && commaPos != std::string::npos && stylePos < commaPos) {
            // Truncate the line before the position of the comma
            line = line.substr(0, commaPos);
        }

        outputFile << line << '\n';
    }

    inputFile.close();
    outputFile.close();

//    std::cout << "Processed " << inputFileName << " and saved as " << outputFileName << std::endl;
}




//
//int main() {
//    processDotFile();
//    insertNewlineBeforeStyle("output.dot","output4.dot");
//    removeBackslashWithSpaceFromPos("output4.dot", "outputFinal.dot");
//}