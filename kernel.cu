#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

#define NUMBER_OF_CUSTOMERS 1000u // How many struct there are in the vector.

#define MAX_USERNAME_LENGTH 20u
#define MAX_BIO_LENGTH 20u

#define HASH_FUNCTION_SIZE 1027u // Size of the output space of the hash function.
#define HASH_SHIFT 6u

#define THREAD_NUMBER 10u // The number of threads you want to use.

#define SAMPLE_FILE_PRINT 1

// USED NAMESPACES
using namespace std;

struct strutturaCustomer {
    char* username; // Identifier field (must be unique for each customer).
    uint64_t number = 0; // Not unique and expected little field.
    char* bio;// Not unique and expected big field.
};

__device__ void gpu_strlen(const char* str, size_t& len) {
    len = 0;
    while (str[len] != '\0') {
        len++;
    }
}

__device__ void bitwise_hash_16(char* str, size_t& size, uint16_t& hash) {
    hash = str[0]; // Il primo valore è il primo carattere della stringa.
    for (uint16_t iter = 1; iter < size; iter++) {
        hash += (hash << HASH_SHIFT) + str[iter];
        // Hash bitwise: shift a sinistra di un certo numero di posizioni e poi aggiungi il carattere corrente
    }
    hash %= HASH_FUNCTION_SIZE; // Il digest deve essere all'interno dell'intervallo di output della funzione hash.
}

__global__ void processCustomers(strutturaCustomer* customers, uint64_t size, strutturaCustomer** res, float* overflowIndexes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t hash;
    size_t len = 0;

    // Ogni thread elabora un subset di elementi nell'array customers
    while (idx < size) {
        gpu_strlen(customers[idx].username, len);
        bitwise_hash_16(customers[idx].username, len, hash);

        int index = atomicAdd(&overflowIndexes[hash], 1);
        res[hash][index] = customers[idx];

        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
}

void cudaMemoryInfo(){
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Memoria libera sulla GPU: " << freeMem / (1024*1024)<< "/" << totalMem / (1024*1024) << std::endl;
}

int main() {
    cout << "Prova partenza del programma." << endl;
    
    uint64_t i = 0, j = 0;
    string username = "";
    strutturaCustomer* h_customers = new strutturaCustomer[NUMBER_OF_CUSTOMERS]; // Array delle strutture dati sulla CPU
    strutturaCustomer** h_res = new strutturaCustomer*[HASH_FUNCTION_SIZE];

    // Allocazione della memoria per i puntatori dentro h_customers
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++) {
        h_customers[i].username = new char[MAX_USERNAME_LENGTH];
        h_customers[i].bio = new char[MAX_BIO_LENGTH];
    }

    // Allocazione della memoria per h_res e i suoi campi
    for(i = 0; i < HASH_FUNCTION_SIZE; i++) {
        h_res[i] = new strutturaCustomer[NUMBER_OF_CUSTOMERS];
        for(j = 0; j < NUMBER_OF_CUSTOMERS; j++) {
            h_res[i][j].username = new char[MAX_USERNAME_LENGTH];
            h_res[i][j].bio = new char[MAX_BIO_LENGTH];
        }
    }

    float* h_overflowIndexes = new float[HASH_FUNCTION_SIZE];

    // Inizializzazione dei dati dei clienti (esempio)
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i) {
        username = "user_" + std::to_string(i);
        strcpy(h_customers[i].username, username.c_str());
        h_customers[i].number = i;
        username = "Bio for user_" + std::to_string(i);
        strcpy(h_customers[i].bio, username.c_str());
    }

    // Inizializzazione degli indici di overflow.
    for (i = 0; i < HASH_FUNCTION_SIZE; i++) {
        h_overflowIndexes[i] = 0.0f;
    }

    float* d_overflowIndexes;
    cudaMalloc((void**)&d_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(float)); // Allocazione della memoria sulla GPU per h_overflowIndexes
    cudaMemcpy(d_overflowIndexes, h_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(float), cudaMemcpyHostToDevice);   // Copia dei dati dalla CPU alla GPU per h_overflowIndexes

    strutturaCustomer* d_customers;
    cudaMalloc((void**)&d_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer));  // Allocazione della memoria sulla GPU per h_customers
    cudaMemcpy(d_customers, h_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_customers

    strutturaCustomer** d_res;  // Creiamo la tabella di hash nella GPU
    cudaMalloc((void**)&d_res, HASH_FUNCTION_SIZE * sizeof(strutturaCustomer*));

    for (i = 0; i < HASH_FUNCTION_SIZE; i++) {
        strutturaCustomer* row;
        cudaMalloc((void**)&row, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer));
        cudaMemcpy(d_res + i, &row, sizeof(strutturaCustomer*), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_customers
    }

    processCustomers<<<NUMBER_OF_CUSTOMERS / THREAD_NUMBER, THREAD_NUMBER>>>(d_customers, NUMBER_OF_CUSTOMERS, d_res, d_overflowIndexes);

    cudaDeviceSynchronize();  // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.
    
    // Copia dei risultati dalla GPU alla CPU
    for (i = 0; i < HASH_FUNCTION_SIZE; i++) {
        strutturaCustomer* row;
        cudaMemcpy(&row, d_res + i, sizeof(strutturaCustomer*), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_res[i], row, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer), cudaMemcpyDeviceToHost);
    }

    cout << "Inizio deallocazione" << endl;

    // DEALLOCAZIONE

    cudaFree(d_customers);   // Deallocazione della memoria sulla GPU per h_customers

    // Deallocazione della memoria sulla GPU per h_res
    for (i = 0; i < HASH_FUNCTION_SIZE; i++) {
        strutturaCustomer* row;
        cudaMemcpy(&row, d_res + i, sizeof(strutturaCustomer*), cudaMemcpyDeviceToHost);
        cudaFree(row);
    }
    cudaFree(d_res);

    // Rilascio della memoria allocata
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++) {
        delete[] h_customers[i].username;
        delete[] h_customers[i].bio;
    }
    delete[] h_customers;

    for (i = 0; i < HASH_FUNCTION_SIZE; i++) {
        for (j = 0; j < NUMBER_OF_CUSTOMERS; j++) {
            delete[] h_res[i][j].username;
            delete[] h_res[i][j].bio;
        }
        delete[] h_res[i];
    }
    delete[] h_res;

    delete[] h_overflowIndexes;

    return 0;
}
