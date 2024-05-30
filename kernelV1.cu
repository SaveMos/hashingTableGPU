// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Used libraries
#include <stdio.h>
#include <chrono> // For the steady-clock.

#include "utilityFile.h" // A utility file, in which we can find configuration constants and file functions.

// Used namespaces
using namespace std;

// Data structure array configuration
#define NUMBER_OF_CUSTOMERS 10000u // How many struct there are in the vector.

#define THREAD_NUMBER_GPU 32u // Number of thread per block.
#define BLOCKS_NUMBER 2000u // Number of blocks.

// Hash function configuration
#define HASH_FUNCTION_SIZE 1027u // Size of the output space of the hash function.

// Other configuration.
#define SAMPLE_FILE_PRINT true // Set 'true' if you want to print the execution time in a '.csv' file.
#define PRINT_CHECKS false // Set 'true' if you want to print the execution time on the prompt.

// Macro per far funzionare le "<<<>>>" in Visual Studio Community.
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

// The target data structure.
struct strutturaCustomer
{
    char *username; // Identifier field (must be unique for each customer).
    uint64_t number = 0; // Not unique and expected little field.
    char *bio;  // Not unique, but it is expected to have a big length.
};

// GPU function to compute the lenght of a string.
__device__ void gpu_strlen(const char *str, size_t &len)
{
    len = 0;
    while (str[len] != '\0') // Quando si raggiunge la marca di fine stringa ci si ferma.
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

__global__ void processCustomers(strutturaCustomer *customers, uint64_t size, strutturaCustomer **res, unsigned int *overflowIndexes)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t len = 0; // Temporary variables for store the username's length.
    uint16_t hash;  // Temporary variables for store the username's 16-bit hash.
    unsigned int index; // The index of the column to write.
    // Ogni thread elabora un subset di elementi nell'array customers
    while (idx < size) // Escape condition.
    {
        gpu_strlen(customers[idx].username, len); // Compute the length of the username.
        bitwise_hash_16(customers[idx].username, len, hash); // Compute the hash of the username.

        index = atomicAdd(&overflowIndexes[hash], 1u); // Atomically increment the index of the column to write.
        res[hash][index] = customers[idx]; // Insert the struct in the table.

        idx += blockDim.x * gridDim.x;
    }
    __syncthreads(); // Wait all the threads of the warp.
}

int main()
{
    uint64_t i = 0, j = 0; // Iterator variables.
    uint64_t count = 0; // Count variables for checking the result.
    string username = "";  // Temporary variable for the inizialization of the customers array.
    strutturaCustomer *h_customers = new strutturaCustomer[NUMBER_OF_CUSTOMERS]; // Array delle strutture dati sulla CPU
    strutturaCustomer **h_res = new strutturaCustomer *[HASH_FUNCTION_SIZE]; // The hash table in the host.
    unsigned int *h_overflowIndexes = new unsigned int [HASH_FUNCTION_SIZE]; // The overflow indexes on the host.
    
    cudaEvent_t tic, toc; // Variables for compute the execution time of the CUDA kernel.
    float elapsed = 0.0f; // Variable for compute the execution time.

    //decltype(std::chrono::steady_clock::now()) start_steady, end_steady; // The definition of the used timer variables.
    
    (cudaEventCreate(&tic));
    (cudaEventCreate(&toc));

    // Allocazione della memoria per i puntatori dentro h_customers
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        h_customers[i].username = new char[MAX_USERNAME_LENGTH];
        h_customers[i].bio = new char[MAX_BIO_LENGTH];
    }

    // Inizializzazione dei dati dei clienti.
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i)
    {
        username = "user_" + to_string(i);
        strcpy(h_customers[i].username, username.c_str());
        h_customers[i].number = i;
        username = "Bio for user_" + to_string(i);
        strcpy(h_customers[i].bio, username.c_str());
    }

    // Allocazione della memoria host per h_res.
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        h_res[i] = new strutturaCustomer[NUMBER_OF_CUSTOMERS];  // Alloco lo spazio per ogni riga della tabella.
        for (j = 0; j < NUMBER_OF_CUSTOMERS; j++)
        {
            h_res[i][j].username = new char[MAX_USERNAME_LENGTH]; // Alloco lo spazio per ogni username.
            h_res[i][j].bio = new char[MAX_BIO_LENGTH];  // Alloco lo spazio per ogni bio.
        }
    }

    // Inizializzazione degli indici di overflow.
    // L'indice di overflow serve per stabilire in modo certo quanti elementi ci sono in ogni riga della tabella.
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        h_overflowIndexes[i] = 0; // All'inizio ogni riga ha 0 elementi.
    }

    //start_steady = std::chrono::steady_clock::now(); // Start measuring the execution time of the main process.

    // Allocazione overflow indexes in GPU.
    unsigned int *d_overflowIndexes;
    cudaMalloc((void **)&d_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(unsigned int));  // Allocazione della memoria sulla GPU per h_overflowIndexes
    cudaMemcpy(d_overflowIndexes, h_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_overflowIndexes

    // Allocazione customers in GPU.
    strutturaCustomer *d_customers;
    cudaMalloc((void **)&d_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer));  // Allocazione della memoria sulla GPU per h_customers
    cudaMemcpy(d_customers, h_customers, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_customers

    // Allocazione delle stringhe all'interno delle strutture dati
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i)
    {      
        // Allocazione dello username nella GPU.
        char *d_username;
        cudaMalloc((void **)&d_username, MAX_USERNAME_LENGTH * sizeof(char));  // Allocazione della memoria per il nome utente sul device
        cudaMemcpy(d_username, h_customers[i].username, MAX_USERNAME_LENGTH * sizeof(char), cudaMemcpyHostToDevice);    // Copia del nome utente dalla CPU alla GPU
        cudaMemcpy(&(d_customers[i].username), &d_username, sizeof(char *), cudaMemcpyHostToDevice); // Aggiornamento del puntatore del nome utente nella struttura dati sul device.    
        
        // Allocazione della bio nella GPU.
        char *d_bio;
        cudaMalloc((void **)&d_bio, MAX_BIO_LENGTH * sizeof(char));  // Allocazione della memoria per la bio sul device.
        cudaMemcpy(d_bio, h_customers[i].bio, MAX_BIO_LENGTH * sizeof(char), cudaMemcpyHostToDevice);  // Copia della bio dalla CPU alla GPU
        cudaMemcpy(&(d_customers[i].bio), &d_bio, sizeof(char *), cudaMemcpyHostToDevice);  // Aggiornamento del puntatore della bio nella struttura dati sul device
    }

    // Allocazione res in GPU.
    strutturaCustomer **d_res; // Creiamo la tabella di hash nella GPU
    cudaMalloc((void **)&d_res, HASH_FUNCTION_SIZE * sizeof(strutturaCustomer *));

    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        strutturaCustomer *row;
        cudaMalloc((void **)&row, NUMBER_OF_CUSTOMERS * sizeof(strutturaCustomer)); // Allocazione della memoria per il vvettore dei customers nella GPU.
        cudaMemcpy(d_res + i, &row, sizeof(strutturaCustomer *), cudaMemcpyHostToDevice); // Copia dei dati dalla CPU alla GPU per h_customers.
    }
    (cudaEventRecord(tic, 0));
    processCustomers KERNEL_ARGS2(BLOCKS_NUMBER, THREAD_NUMBER_GPU) (d_customers, NUMBER_OF_CUSTOMERS, d_res, d_overflowIndexes);

     (cudaEventRecord(toc, 0));

    (cudaDeviceSynchronize()); // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.

    (cudaEventSynchronize(toc)); // Synchronize the event.

    (cudaEventElapsedTime(&elapsed, tic, toc)); // Compute the elapsed 
    cudaMemcpy(h_overflowIndexes, d_overflowIndexes, HASH_FUNCTION_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);  // Copia overflowIndexes dalla GPU alla CPU.

    // Copia della tabella di hash (res) dalla GPU alla CPU.
    char *username_host, *bio_host;
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        if(h_overflowIndexes[i] == 0){
            continue; // No elements in this row, go on.
        }

        strutturaCustomer *row;
        cudaMemcpy(&row, d_res + i, sizeof(strutturaCustomer *), cudaMemcpyDeviceToHost); // copio l'indirizzo della row,
        cudaMemcpy(h_res[i], row, h_overflowIndexes[i] * sizeof(strutturaCustomer), cudaMemcpyDeviceToHost); // Scarico il contenuto della riga.

        // Copy strings for each strutturaCustomer.
        for (j = 0; j < h_overflowIndexes[i]; j++)
        {
            h_res[i][j].username = new char[MAX_USERNAME_LENGTH]; // Allocazione della memoria per il nome utente sulla CPU.
            h_res[i][j].bio = new char[MAX_BIO_LENGTH]; // Allocazione della memoria per la bio sulla CPU.

            cudaMemcpy(&username_host, &(row[j].username), sizeof(char *), cudaMemcpyDeviceToHost);  // Copia del puntatore dello username dalla GPU all'host.
            cudaMemcpy(h_res[i][j].username, username_host, MAX_USERNAME_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);   // Copia dello username dalla GPU alla CPU.

            cudaMemcpy(&bio_host, &(row[j].bio), sizeof(char *), cudaMemcpyDeviceToHost); // Copia del puntatore della bio dalla GPU all'host.
            cudaMemcpy(h_res[i][j].bio, bio_host, MAX_BIO_LENGTH * sizeof(char), cudaMemcpyDeviceToHost); // Copia dei dati della bio dalla GPU alla CPU
        }
    }

    //end_steady = std::chrono::steady_clock::now(); // Measure the execution time of the main process when all the threads are ended.
	//std::chrono::duration<double> elapsed_seconds_high_res = end_steady - start_steady; // Compute the execution time.
	//double time = elapsed_seconds_high_res.count(); // Return the total execution time.

    if (PRINT_CHECKS)
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

    cudaFree(d_customers); // Deallocazione della memoria sulla GPU per d_customers

    // Deallocazione della memoria sulla GPU per d_res
    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        // Dealloco ogni riga di d_res dalla GPU.
        strutturaCustomer *row;
        cudaMemcpy(&row, d_res + i, sizeof(strutturaCustomer *), cudaMemcpyDeviceToHost);
        cudaFree(row);
    }
    cudaFree(d_res); // Dealloco d_res dalla GPU.

    // Rilascio della memoria allocata della CPU.
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        delete[] h_customers[i].username; // Dealloco lo username della struttura i.
        delete[] h_customers[i].bio; // Dealloco la bio della struttura i.
    }
    delete[] h_customers; // Dealloco il vettore delle strutture dati nella CPU.

    for (i = 0; i < HASH_FUNCTION_SIZE; i++)
    {
        for (j = 0; j < NUMBER_OF_CUSTOMERS; j++)
        {
            delete[] h_res[i][j].username; // Dealloco lo username della struttura a riga i e colonna j.
            delete[] h_res[i][j].bio; // Dealloco la bio della struttura a riga i e colonna j.
        }
        delete[] h_res[i]; // Dealloco la riga i della tabella di hash.
    }
    delete[] h_res; // Dealloco ila tabella di hash nella CPU.
    delete[] h_overflowIndexes; // Dealloco il vettore degli indici nella CPU.
    
    if (SAMPLE_FILE_PRINT)
    {
        // elapsed must be float, but the function wants double
        printToFile(static_cast<double>(elapsed), "kernelV1.csv"); // Print the sample in the '.csv' file.
        insertNewLine("kernelV1.csv");
        //printToFile(time, "total.csv"); // Print the sample in the '.csv' file.
        //insertNewLine("total.csv");
    }

    return 0;
}
