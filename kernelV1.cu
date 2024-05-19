// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Used libraries
#include <iostream> // To remove after testing.
#include <stdio.h>
#include <vector>
#include <string>
#include <chrono>

#include "utilityFile.h"


// Data structure array configuration
#define NUMBER_OF_CUSTOMERS 1000u // How many struct there are in the vector.

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
    char *username;      // Identifier field (must be unique for each customer).
    uint64_t number = 0; // Not unique and expected little field.
    char *bio;           // Not unique and expected big field.
};

#define MAX_SEMAPHORE_VALUE 1

__device__ int mutexVector[HASH_FUNCTION_SIZE]; // Array di semafori

__device__ void lock(int *semaphore)
{
    while (atomicCAS(semaphore, 0, 1) != 0);
    __threadfence();  
}

__device__ void unlock(int *semaphore)
{
   __threadfence();  
    atomicExch(semaphore, 0);
}

// GPU function to compute the lenght of a string.
__device__ void gpu_strlen(const char *str, size_t &len)
{
    len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
}

// GPU function for compute the 16-bit hash of a string.
__device__ void bitwise_hash_16(const char *str, size_t size, uint16_t &hash)
{
    hash = str[0];
    for (size_t iter = 1; iter < size; iter++)
    {
        hash = (hash << HASH_SHIFT) ^ str[iter];
    }
    hash %= HASH_FUNCTION_SIZE; // Il digest deve essere all'interno dell'intervallo di output della funzione hash.
}

__global__ void processCustomers(strutturaCustomer *customers, uint64_t size, strutturaCustomer **res, int *overflowIndexes)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x, index;
    size_t len = 0;
    uint16_t hash;
    // Ogni thread elabora un subset di elementi nell'array customers
    while (idx < size)
    {
        gpu_strlen(customers[idx].username, len);
        bitwise_hash_16(customers[idx].username, len, hash);

        lock(&mutexVector[hash]);
        res[hash][static_cast<int>(overflowIndexes[hash])] = customers[idx];
        overflowIndexes[hash]++;
        // printf("Username: %s | lenght: %u | Hash: %i | OI: %f\n" , customers[idx].username , len , hash, overflowIndexes[hash]);
        unlock(&mutexVector[hash]);

        idx += blockDim.x * gridDim.x;
    }

    __syncthreads();
}

void cudaMemoryInfo()
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Memoria libera sulla GPU: " << freeMem / (1024 * 1024) << "/" << totalMem / (1024 * 1024) << std::endl;
}

int main()
{
    cout << "Prova partenza del programma." << endl;

    uint64_t i = 0, j = 0, count = 0;
    string username = "";                                                        // Temporary variable for the inizialization of the customers array.
    strutturaCustomer *h_customers = new strutturaCustomer[NUMBER_OF_CUSTOMERS]; // Array delle strutture dati sulla CPU
    strutturaCustomer **h_res = new strutturaCustomer *[HASH_FUNCTION_SIZE];
    int *h_overflowIndexes = new int[HASH_FUNCTION_SIZE];

    //variable for elapsed time
    cudaEvent_t tic, toc;
    float elapsed;
    
    decltype(std::chrono::steady_clock::now()) start_steady, end_steady; // The definition of the used timer variables.

    cudaEventCreate(&tic);
    cudaEventCreate(&toc);

    // Allocazione della memoria per i puntatori dentro h_customers
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        h_customers[i].username = new char[MAX_USERNAME_LENGTH];
        h_customers[i].bio = new char[MAX_BIO_LENGTH];
    }

    // Inizializzazione dei dati dei clienti (esempio)
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i)
    {
        username = "user_" + to_string(i);
        strcpy(h_customers[i].username, username.c_str());
        h_customers[i].number = i;
        username = "Bio for user_" + to_string(i);
        strcpy(h_customers[i].bio, username.c_str());
    }

    // Allocazione della memoria per h_res e i suoi campi
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        h_res[i] = new strutturaCustomer[NUMBER_OF_CUSTOMERS];
        for (j = 0; j < NUMBER_OF_CUSTOMERS; j++)
        {
            h_res[i][j].username = new char[MAX_USERNAME_LENGTH];
            h_res[i][j].bio = new char[MAX_BIO_LENGTH];
        }
    }

    // Inizializzazione degli indici di overflow.
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        h_overflowIndexes[i] = 0.0f;
    }


    start_steady = std::chrono::steady_clock::now(); // Start measuring the execution time of the main process.

    // Allocazione overflow indexes in GPU.
    int *d_overflowIndexes;
    cudaMalloc((void **)&d_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(int));                                  // Allocazione della memoria sulla GPU per h_overflowIndexes
    cudaMemcpy(d_overflowIndexes, h_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(int), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_overflowIndexes

    // Allocazione customers in GPU.
    strutturaCustomer *d_customers;
    cudaMalloc((void **)&d_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer));                            // Allocazione della memoria sulla GPU per h_customers
    cudaMemcpy(d_customers, h_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_customers

    // Allocazione delle stringhe all'interno delle strutture dati
    // Allocazione delle stringhe all'interno delle strutture dati
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i)
    {
        // Allocazione della memoria per il nome utente sul device
        char *d_username;
        cudaMalloc((void **)&d_username, MAX_USERNAME_LENGTH * sizeof(char));
        // Copia del nome utente dalla CPU alla GPU
        cudaMemcpy(d_username, h_customers[i].username, MAX_USERNAME_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
        // Aggiornamento del puntatore del nome utente nella struttura dati sul device
        cudaMemcpy(&(d_customers[i].username), &d_username, sizeof(char *), cudaMemcpyHostToDevice);

        // Allocazione della memoria per la bio sul device
        char *d_bio;
        cudaMalloc((void **)&d_bio, MAX_BIO_LENGTH * sizeof(char));
        // Copia della bio dalla CPU alla GPU
        cudaMemcpy(d_bio, h_customers[i].bio, MAX_BIO_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
        // Aggiornamento del puntatore della bio nella struttura dati sul device
        cudaMemcpy(&(d_customers[i].bio), &d_bio, sizeof(char *), cudaMemcpyHostToDevice);
    }

    // Allocazione res in GPU.
    strutturaCustomer **d_res; // Creiamo la tabella di hash nella GPU
    cudaMalloc((void **)&d_res, HASH_FUNCTION_SIZE * sizeof(strutturaCustomer *));

    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        strutturaCustomer *row;
        cudaMalloc((void **)&row, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer));
        cudaMemcpy(d_res + i, &row, sizeof(strutturaCustomer *), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_customers
    }
    cout << "Inizio nucleo" << endl;

    cudaEventRecord(tic, 0); 
    processCustomers<<<NUMBER_OF_CUSTOMERS / THREAD_NUMBER, THREAD_NUMBER>>>(d_customers, NUMBER_OF_CUSTOMERS, d_res, d_overflowIndexes);
    cudaEventRecord(toc, 0);

    cout << "fine nucleo" << endl;
    //synchronize the event
    cudaEventSynchronize(toc);

    // processCustomers<<<1, 1>>>(d_customers, NUMBER_OF_CUSTOMERS, d_res, d_overflowIndexes);

    cudaDeviceSynchronize(); // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.

    //compute the elapsed time
    cudaEventElapsedTime(&elapsed, tic, toc);

    cout << "Inizio copia dei risultati" << endl;

    // Copia overflowIndexes dalla GPU alla CPU
    // In modo da sapere quanti elementi ci sono per ogni riga.
    cudaMemcpy(h_overflowIndexes, d_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Copia dei risultati dalla GPU alla CPU
    char *username_host, *bio_host;
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        if(h_overflowIndexes[i] == 0){
            continue;
        }
        strutturaCustomer *row;
        cudaMemcpy(&row, d_res + i, sizeof(strutturaCustomer *), cudaMemcpyDeviceToHost); // copio indirizzo su row
        cudaMemcpy(h_res[i], row, h_overflowIndexes[i] * sizeof(strutturaCustomer), cudaMemcpyDeviceToHost);

        // Copy strings for each strutturaCustomer.
        for (j = 0; j < h_overflowIndexes[i]; j++)
        {
            // Copia del puntatore del nome utente dalla GPU all'host
            cudaMemcpy(&username_host, &(row[j].username), sizeof(char *), cudaMemcpyDeviceToHost);
            // Allocazione della memoria per il nome utente sulla CPU
            h_res[i][j].username = new char[MAX_USERNAME_LENGTH];

            // Copia dei dati del nome utente dalla GPU alla CPU
            cudaMemcpy(h_res[i][j].username, username_host, MAX_USERNAME_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);

            // Copia del puntatore della bio dalla GPU all'host
            cudaMemcpy(&bio_host, &(row[j].bio), sizeof(char *), cudaMemcpyDeviceToHost);

            // Allocazione della memoria per la bio sulla CPU
            h_res[i][j].bio = new char[MAX_BIO_LENGTH];

            // Copia dei dati della bio dalla GPU alla CPU
            cudaMemcpy(h_res[i][j].bio, bio_host, MAX_BIO_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);
        }
    }

    if (PRINT_CHECKS)
    {
        for (i = 0; i < HASH_FUNCTION_SIZE; i++)
        {
            if(h_overflowIndexes[i] == 0){
                continue;
            }
            cout << i << ") ";
            for (j = 0; j < h_overflowIndexes[i]; j++)
            {
                if (strlen(h_res[i][j].username) == 0)
                {
                    break;
                }
                cout << h_res[i][j].username << " => ";
            }
            cout << endl;
        }
    }

    end_steady = std::chrono::steady_clock::now(); // Measure the execution time of the main process when all the threads are ended.
	std::chrono::duration<double> elapsed_seconds_high_res = end_steady - start_steady; // Compute the execution time.
	double time = elapsed_seconds_high_res.count(); // Return the total execution time.

    if (CHECKS)
    {
        for (i = 0; i < HASH_FUNCTION_SIZE; i++)
        {
            for (j = 0; j < h_overflowIndexes[i]; j++)
            {
                if (strlen(h_res[i][j].username) == 0)
                {
                    break;
                }
                count++;
            }
        }


        if (count == NUMBER_OF_CUSTOMERS)
        {
            cout << count << " OK" << endl;
        }
        else
        {
            cout << count << " NOT OK" << endl;
        }
    }

    cout << "Inizio deallocazione" << endl;

    // DEALLOCAZIONE

    cudaFree(d_customers); // Deallocazione della memoria sulla GPU per h_customers

    // Deallocazione della memoria sulla GPU per h_res
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        strutturaCustomer *row;
        cudaMemcpy(&row, d_res + i, sizeof(strutturaCustomer *), cudaMemcpyDeviceToHost);
        cudaFree(row);
    }
    cudaFree(d_res);

    //free the two events tic and toc
    cudaEventDestroy(tic);
    cudaEventDestroy(toc);

    // Rilascio della memoria allocata
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        delete[] h_customers[i].username;
        delete[] h_customers[i].bio;
    }
    delete[] h_customers;

    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        for (j = 0; j < NUMBER_OF_CUSTOMERS; j++)
        {
            delete[] h_res[i][j].username;
            delete[] h_res[i][j].bio;
        }
        delete[] h_res[i];
    }
    delete[] h_res;

    delete[] h_overflowIndexes;

    cout<< "tempo esecuzione: "<< elapsed<<endl;
    //elapsed must be float, but the function wants double
    double elapsed1 = static_cast<double>(elapsed);
    printToFile(elapsed1, "kernel1.csv"); // Print the sample in the '.csv' file.
    insertNewLine("kernel1.csv");
    cout << "Tempo di esecuzione totale : " << time << " s" << endl;
    printToFile(time, "total1.csv"); // Print the sample in the '.csv' file.
    insertNewLine("total1.csv");

    return 0;
}
