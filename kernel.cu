// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Used libraries
#include <iostream> // To remove after testing.
#include <stdio.h>
#include <vector>
#include <string>

// Data structure array configuration
#define NUMBER_OF_CUSTOMERS 10000u // How many struct there are in the vector.

// Parallelism configuration
#define THREAD_NUMBER 10u // The number of threads you want to use.

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
    __syncthreads();
}

int main()
{
    uint64_t i = 0, count = 0;

    vector<strutturaCustomer> customers; // The list of the customers.
    vector<vector<strutturaCustomer>> ret(HASH_FUNCTION_SIZE); // The final hashing table.

    uint16_t *hashes = new uint16_t[NUMBER_OF_CUSTOMERS];
    char **h_customers = new char *[NUMBER_OF_CUSTOMERS];

    cudaEvent_t tic, toc; // Variables for compute the elapsed time.
    float elapsed = 0.0f; // Variable for compute the elapsed time.

    strutturaCustomer str;
    string username = "";  // Temporary variable for the inizialization of the customers array.

    cout << "Programma partito!" << endl;
    cout << endl;
    cout << endl;

    cudaEventCreate(&tic);
    cudaEventCreate(&toc);

  
    // Inizializzazione dei dati dei clienti (esempio)
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i)
    {
        str.username = "user_" + to_string(i);
        str.number = i;
        str.bio = "Bio for user_" + to_string(i);

        h_customers[i] = new char[str.username.length()];
        strcpy(h_customers[i], str.username.c_str());
        customers.push_back(str);

        hashes[i] = 0;
    }

    cout << "Inizializzazione delle strutture dati..." << endl;
    // Allocazione overflow indexes in GPU.
    uint16_t *d_hashes;
    cudaMalloc((void **)&d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t)); // Allocazione della memoria sulla GPU per h_overflowIndexes
    //cudaMemcpy(d_hashes, hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_overflowIndexes

    cout << "Vettore hashes generato e allocato in GPU!" << endl;
    // Allocazione customers in GPU.
    char **d_customers; // Creiamo la tabella di hash nella GPU
    cudaMalloc((void **)&d_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer *));
    size_t size_str;
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        size_str = customers[i].username.length();
        char *d_username;
        cudaMalloc((void **)&d_username, size_str * sizeof(char));
        // Copia del nome utente dalla CPU alla GPU
        cudaMemcpy(d_username, h_customers[i], size_str * sizeof(char), cudaMemcpyHostToDevice);
        // Aggiornamento del puntatore del nome utente nella struttura dati sul device
        cudaMemcpy(&(d_customers[i]), &d_username, sizeof(char *), cudaMemcpyHostToDevice);
    }
    cout << "Vettore customers generato e allocato in GPU!" << endl;
    cout << endl;
    cout << endl;

    cout << "Inizio del nucleo." << endl;

    cudaEventRecord(tic, 0);
    processCustomers<<<NUMBER_OF_CUSTOMERS / THREAD_NUMBER, THREAD_NUMBER>>>(d_customers, NUMBER_OF_CUSTOMERS, d_hashes);
    cudaEventRecord(toc, 0);

    cout << "Fine del nucleo." << endl;
    cout << endl;
    cout << endl;

     
    cudaDeviceSynchronize(); // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.

    cudaEventSynchronize(toc); // synchronize the event
   
    cudaEventElapsedTime(&elapsed, tic, toc); // Compute the elapsed time

    cout << "Copia dei risultati..." << endl;
    cudaMemcpy(hashes, d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t), cudaMemcpyDeviceToHost); // Copia dei risultati dalla GPU alla CPU.
    cout << "Risultati copiati in memoria!" << endl;

    // Costruzione della tabella di hashing.
    cout << "Costruzione della tabella di hash..." << endl;
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        ret[hashes[i]].push_back(customers[i]);
    }
    cout << "Tabella di Hash";
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        count += ret[i].size();
    }

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

    // DEALLOCAZIONE

    cudaFree(d_customers); // Deallocazione della memoria sulla GPU per d_customers.
    cudaFree(d_hashes);    // Deallocazione della memoria sulla GPU per d_hashes.
    cout << "Deallocazione GPU completata!" << endl;

    // Rilascio della memoria CPU allocata
    delete[] h_customers;
    delete[] hashes;
    customers.clear(); // Polizia del vettore originale.

    cout << "Deallocazione CPU completata!" << endl;

    // Free the two events tic and toc
    cudaEventDestroy(tic);
    cudaEventDestroy(toc);
    cout << "Deallocazione Eventi Timer completata!" << endl;

    cout << "-----------------------------------------" << endl;
    cout << "Tempo di esecuzione del nucleo: " << elapsed << endl;
    cout << "-----------------------------------------" << endl;

    return 0;
}
