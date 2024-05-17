// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Used libraries
#include <iostream> // To remove after testing.
#include <stdio.h>
#include <vector>
#include <string>

// Data structure array configuration
#define NUMBER_OF_CUSTOMERS 1000000u // How many struct there are in the vector.

// Parallelism configuration
#define THREAD_NUMBER 1u // The number of threads you want to use.

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
    cout << "Prova partenza del programma." << endl;

    uint64_t i = 0, count = 0;
    string username = "";                                      // Temporary variable for the inizialization of the customers array.
    vector<strutturaCustomer> customers;                       // The list of the customers.
    vector<vector<strutturaCustomer>> ret(HASH_FUNCTION_SIZE); // The final hashing table.
    
    uint16_t *hashes = new uint16_t[NUMBER_OF_CUSTOMERS];
    char **h_customers = new char* [NUMBER_OF_CUSTOMERS];

    strutturaCustomer str;
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

    cout << "strutture inizializzate" << endl;
    // Allocazione overflow indexes in GPU.
    uint16_t *d_hashes;
    cudaMalloc((void **)&d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t)); // Allocazione della memoria sulla GPU per h_overflowIndexes
    //cudaMemcpy(d_hashes, hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_overflowIndexes

    cout << "hashes ok" << endl;
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
    cout << "Inizio nucleo." << endl;

    processCustomers<<<NUMBER_OF_CUSTOMERS / THREAD_NUMBER, THREAD_NUMBER>>>(d_customers, NUMBER_OF_CUSTOMERS, d_hashes);

    cout << "fine nucleo." << endl;

    cudaDeviceSynchronize(); // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.

    cout << "Inizio copia dei risultati." << endl;

    // Copia overflowIndexes dalla GPU alla CPU
    // In modo da sapere quanti elementi ci sono per ogni riga.
    cudaMemcpy(hashes, d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        ret[hashes[i]].push_back(customers[i]);
    }
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        count += ret[i].size();
    }

    if (count == NUMBER_OF_CUSTOMERS)
    {
        cout << count << " OK" << endl;
    }
    else
    {
        cout << count << " NOT OK" << endl;
    }

    // Copia dei risultati dalla GPU alla CPU

    cout << "Inizio deallocazione." << endl;

    // DEALLOCAZIONE

    cudaFree(d_customers); // Deallocazione della memoria sulla GPU per h_customers
    cudaFree(d_hashes);

    // Rilascio della memoria allocata
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        delete[] h_customers[i];
    }
    delete[] h_customers;

    delete[] hashes;

    return 0;
}
