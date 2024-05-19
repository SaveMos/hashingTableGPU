// Data stream libraries.
#include <fstream> // Library for writing the samples in the '.csv' file.
#include <iostream>
#include <string>



// Parallelism configuration
#define THREAD_NUMBER_CPU 8u // The number of threads you want to use.

#define HASH_SHIFT 6u

// Other configuration
#define SAMPLE_FILE_PRINT 1
#define CHECKS 1
#define PRINT_CHECKS 1

#define MAX_USERNAME_LENGTH 20
#define MAX_BIO_LENGTH 20

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