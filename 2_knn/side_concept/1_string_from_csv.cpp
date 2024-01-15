#include <iostream>
#include<string>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

int main() {
    // This creates an ifstream object named inputFile and associates it with the file named "filename.txt". If the file does not exist, or if there are issues opening the file, the stream will be in a fail state.
    std::ifstream file("file.txt");
    std::string line;

    // read a line of text until a specified delimiter (such as a newline character '\n') is encountered.
    while (std::getline(file, line)) {
        // allows you to treat a string as a stream, enabling you to perform input operations on it
        std::istringstream iss(line);
        std::vector<std::string> tokens;

        while (std::getline(iss, line, ',')) {
            tokens.push_back(line);
        }

        // Now 'tokens' contains the values from the current line
        // and you can handle them according to their data type.
        // Convert strings to numeric types if needed.
        for (string v : tokens)
        {
            cout<<v<<" ";
        }
    }
    if(!std::getline(file, line))
    {
        cout<<"unable to read from the file"<<"\n\n";
    }


    return 0;
}