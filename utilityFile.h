// Data stream libraries.
#include <fstream> // Library for writing the samples in the '.csv' file.
#include <iostream>
#include <string>

// USED NAMESPACES
using namespace std;

void printToFile(double sample, string fileType) {
	std::ofstream file(fileType, std::ios_base::app); // Open the samples '.csv' file.
	string d_str = to_string(sample); // Convert to string the execution time sample.
	size_t pos = d_str.find('.'); // Locate the '.' in the sample string.
	if (pos != std::string::npos) {
		d_str.replace(pos, 1, ","); // Replace the '.' with the ',' for a better understanding from Excell.
	}
	file << d_str << endl; // Write the sample in the .csv file
	file.close(); // Close the samples file
}

void insertNewLine(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Errore nell'apertura del file: " << filename << std::endl;
        return;
    }

    std::string line;
    int counter = 0;

    while (std::getline(inFile, line)) {
        if (line != "-----------------------------------------") {
            ++counter;
        }
    }

    inFile.close();

    if (counter % 30 == 0) {
        std::ofstream outFile(filename, std::ios_base::app);
        if (outFile.is_open()) {
            outFile << "-----------------------------------------" << std::endl;
            outFile.close();
        } else {
            std::cerr << "Errore nell'apertura del file in modalità append." << std::endl;
        }
    }
}