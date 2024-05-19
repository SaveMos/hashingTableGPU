// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Used libraries
#include <iostream> // To remove after testing.
#include <stdio.h>
#include <vector>
#include <string>
#include <chrono> // Library for the steady clock.
#include <thread>

// Data structure array configuration
#define NUMBER_OF_CUSTOMERS 100u // How many struct there are in the vector.

// Parallelism configuration
#define THREAD_NUMBER_CPU 8u // The number of threads you want to use.
#define THREAD_NUMBER 256u

// Data structure configuration
#define MAX_USERNAME_LENGTH 20u
#define MAX_BIO_LENGTH 20u

// Hash configuration
#define HASH_FUNCTION_SIZE 1027u // Size of the output space of the hash function.
#define HASH_SHIFT 6u

// Other configuration
#define SAMPLE_FILE_PRINT 1
#define CHECKS 1
#define PRINT_CHECKS 0

// Used namespaces
using namespace std;

// The target data structure.
struct strutturaCustomer
{
    string username;     // Identifier field (must be unique for each customer).
    uint64_t number = 0; // Not unique and expected little field.
    string bio;          // Not unique and expected big field.
};

__device__ void gpu_strlen(char *str, size_t &len)
{
    len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
}

// GPU function for compute the 16-bit hash of a string.
__device__ void bitwise_hash_16(char *str, size_t size, uint16_t &hash)
{
    hash = str[0];
    for (size_t iter = 1; iter < size; iter++)
    {
        hash = (hash << HASH_SHIFT) ^ str[iter];
    }
    hash %= HASH_FUNCTION_SIZE; // Il digest deve essere all'interno dell'intervallo di output della funzione hash.
}

__global__ void processCustomers(char **customers, uint64_t size, uint16_t *hashes)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t len = 0;
    uint16_t hash;
    // Ogni thread elabora un subset di elementi nell'array customers
    while (idx < size)
    {
        gpu_strlen(customers[idx], len);
        bitwise_hash_16(customers[idx], len, hash);
        hashes[idx] = hash;
        idx += blockDim.x * gridDim.x;
    }
}

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(1);                                                            \
    } }

void threadCodeBuildTableV2(vector<strutturaCustomer>& customers, char** h_customers , uint8_t id) {
	strutturaCustomer c = { "a" , 0 , "Insert his bio"}; // Temporary Customer structure, recycled some times in the code to increase the chance of cache hit.
	string username; // Temporary variable, recycled some times in the code to increase the chance of cache hit
	uint64_t target = id;
	const uint64_t size = customers.size();

	while (target < size) {
		c.username = "user_" + to_string(target);
		customers.at(target) = c; // Insert the user in the list.
	
        h_customers[target] = new char[c.username.length() + 1]; // Aggiunto 1 per il terminatore null.
        strcpy(h_customers[target], c.username.c_str());

        target += THREAD_NUMBER_CPU;
	}
}


int main()
{
    uint64_t i = 0, count = 0;

    vector<strutturaCustomer> customers (NUMBER_OF_CUSTOMERS); // The list of the customers.
    vector<vector<strutturaCustomer>> ret(HASH_FUNCTION_SIZE); // The final hashing table.

    uint16_t *hashes = new uint16_t[NUMBER_OF_CUSTOMERS];
    char **h_customers = new char *[NUMBER_OF_CUSTOMERS];

    cudaEvent_t tic, toc; // Variables for compute the elapsed time.
    float elapsed = 0.0f; // Variable for compute the elapsed time.

    decltype(std::chrono::steady_clock::now()) start_steady, end_steady; // The definition of the used timer variables.

    (cudaEventCreate(&tic));
    (cudaEventCreate(&toc));

    // Inizializzazione dei dati dei clienti (esempio)
   	vector<thread> threadMixer(THREAD_NUMBER_CPU); // Vector of the threads descriptors.
	uint8_t ithread; // Iterator variable.
	uint64_t customersSize = customers.size();

	for (ithread = 0; ithread < THREAD_NUMBER_CPU - 1; ithread++) { // For each started thread...
		thread thread_i(
			threadCodeBuildTableV2, // The thread function.
			ref(customers), // The customers array.
            ref(h_customers),
			ithread // The thread's id.
		);
		threadMixer.at(ithread) = move(thread_i); // Add the thread descriptor to the thread descriptor vector.
	}

	// The main thread too contribute to the generation of the data stucture.
	threadCodeBuildTableV2(
		ref(customers), // The thread function.
        ref(h_customers),
		THREAD_NUMBER_CPU - 1// The thread's id.
	);

    // Now the father wait for all the started threads to finish their execution.
	for (ithread = 0; ithread < THREAD_NUMBER_CPU - 1; ithread++) {
		threadMixer[ithread].join(); // Join the i� thread.
	}

	start_steady = std::chrono::steady_clock::now(); // Start measuring the execution time of the main process.
    
    //cout << "Inizializzazione delle strutture dati..." << endl;
    // Allocazione overflow indexes in GPU.
    uint16_t *d_hashes;
    (cudaMalloc((void **)&d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t))); // Allocazione della memoria sulla GPU per h_overflowIndexes

    //cout << "Vettore hashes generato e allocato in GPU!" << endl;
    // Allocazione customers in GPU.
    char **d_customers; // Creiamo la tabella di hash nella GPU
    uint16_t size_str;
    char *d_username;
    (cudaMalloc((void **)&d_customers, NUMBER_OF_CUSTOMERS * sizeof(char *)));
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        size_str = customers[i].username.length() + 1;
        (cudaMalloc((void **)&d_username, size_str * sizeof(char))); // Copia del nome utente dalla CPU alla GPU
        (cudaMemcpy(d_username, h_customers[i], size_str * sizeof(char), cudaMemcpyHostToDevice)); // Aggiornamento del puntatore del nome utente nella struttura dati sul device
        (cudaMemcpy(&(d_customers[i]), &d_username, sizeof(char*), cudaMemcpyHostToDevice));
    }
    /*
    cout << "Vettore customers generato e allocato in GPU!" << endl;
    cout << endl;
    cout << endl;

    cout << "Inizio del nucleo." << endl;
    */
    (cudaEventRecord(tic, 0));
    processCustomers<<<NUMBER_OF_CUSTOMERS / THREAD_NUMBER, THREAD_NUMBER>>>(d_customers, NUMBER_OF_CUSTOMERS, d_hashes);
    (cudaEventRecord(toc, 0));

    (cudaDeviceSynchronize()); // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.

    (cudaEventSynchronize(toc)); // synchronize the event

    (cudaEventElapsedTime(&elapsed, tic, toc)); // Compute the elapsed time

    /*
    cout << "Fine del nucleo." << endl;
    cout << endl;
    cout << endl;

    cout << "Copia dei risultati..." << endl;
    */
    (cudaMemcpy(hashes, d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t), cudaMemcpyDeviceToHost)); // Copia dei risultati dalla GPU alla CPU.
    //cout << "Risultati copiati in memoria!" << endl;

    // Costruzione della tabella di hashing.
    //cout << "Costruzione della tabella di hash..." << endl;
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        ret[hashes[i]].push_back(customers[i]);
    }
    //cout << "Tabella di Hash";
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        count += ret[i].size();
    }

    end_steady = std::chrono::steady_clock::now(); // Measure the execution time of the main process when all the threads are ended.
	std::chrono::duration<double> elapsed_seconds_high_res = end_steady - start_steady; // Compute the execution time.
	const double time = elapsed_seconds_high_res.count(); // Return the total execution time.
    /*
    if (count == NUMBER_OF_CUSTOMERS)
    {
        cout << " costruita con successo!" << endl;
    }
    else
    {
        cout << " non costruita, errore!" << endl;
    }
   
    cout << endl;
    cout << endl;
    cout << "Inizio deallocazione..." << endl;
    */
    // DEALLOCAZIONE
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i) {
        char* d_username;
        (cudaMemcpy(&d_username, &d_customers[i], sizeof(char*), cudaMemcpyDeviceToHost));
        (cudaFree(d_username));
    }

    (cudaFree(d_customers)); // Deallocazione della memoria sulla GPU per d_customers.
    (cudaFree(d_hashes));    // Deallocazione della memoria sulla GPU per d_hashes.
    //cout << "Deallocazione GPU completata!" << endl;

    // Rilascio della memoria CPU allocata
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i) {
        delete[] h_customers[i];
    }
    delete[] h_customers;
    delete[] hashes;
    customers.clear(); // Polizia del vettore originale.

    &&cout << "Deallocazione CPU completata!" << endl;

    // Free the two events tic and toc
    (cudaEventDestroy(tic));
    (cudaEventDestroy(toc));
    cout << "Deallocazione Eventi Timer completata!" << endl;

    cout << "-----------------------------------------" << endl;
    cout << "Tempo di esecuzione del nucleo: " << elapsed << " ms" << endl;
    cout << "Tempo di esecuzione totale : " << time << " s" << endl;
    cout << "-----------------------------------------" << endl;

    return 0;
}
