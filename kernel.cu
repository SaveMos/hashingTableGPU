// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Used libraries
#include <stdio.h>
#include <vector>
#include <chrono> // Library for the steady-clock.
#include <thread>
#include <mutex>
#include <array>

#include "utilityFile.h"

// Used namespaces
using namespace std;

// Data structure array configuration
#define NUMBER_OF_CUSTOMERS 10000u // How many struct there are in the dataset.

// Hash function configuration
#define HASH_FUNCTION_SIZE 1027u // Size of the output space of the hash function.

// GPU Multithreading configuration
#define THREAD_NUMBER_GPU 64u // Number of thread per block.
#define BLOCKS_NUMBER 2000u // Number of blocks.

// Other configuration.
#define SAMPLE_FILE_PRINT true // Set 'true' if you want to print the execution time in a '.csv' file.
#define PRINT_CHECKS false // Set 'true' if you want to print the execution time on the prompt.

struct sharedMutexMixer {
    array<mutex, HASH_FUNCTION_SIZE> mutexes; // An array of mutex.
};

// The target data structure.
struct strutturaCustomer
{
    string username = ""; // Identifier field (must be unique for each customer).
    uint64_t number = 0; // Not unique and expected little field.
    string bio = ""; // Not unique and expected big length.
};

sharedMutexMixer mutexVector; // Global mutex vector, used to grant syncronization during an access to a row of the hash table.

__constant__ uint64_t d_size;

// GPU function for compute the 16-bit hash of a string.
__device__ uint16_t bitwise_hash_16(char* str) {
    uint16_t hash = HASH_FUNCTION_SIZE; // Initial value of the hash.
    uint16_t c = 0;
    while ((c = *str++)) {
        hash = ((hash << 4) ^ hash) + c;
    }
    hash %= HASH_FUNCTION_SIZE; // The hash must be limited by the hash function output size.
    return hash;
}

__global__ void processCustomers(char **customers,  uint16_t *hashes)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Ogni thread elabora un subset di elementi nell'array customers
    while (idx < d_size)
    {
        hashes[idx] = bitwise_hash_16(customers[idx]);
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
}

// Macro per controllare eventuali errori nella GPU.
#define CUDA_CHECK_RETURN(value)                                          \
    {                                                                     \
        cudaError_t _m_cudaStat = value;                                  \
        if (_m_cudaStat != cudaSuccess)                                   \
        {                                                                 \
            fprintf(stderr, "Error %s at line %d in file %s\n",           \
                    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
            exit(1);                                                      \
        }                                                                 \
    }

// Macro per far funzionare le "<<<>>>"
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

void threadCodeInitialize(vector<strutturaCustomer> &customers, char **h_customers, uint8_t id)
{
    strutturaCustomer c = {"", 0, "Insert his bio"}; // Temporary Customer structure, recycled some times in the code to increase the chance of cache hit.
    string username; // Temporary variable, recycled some times in the code to increase the chance of cache hit
    uint64_t target = id; // Starting point of this CPU-Thread.
    const uint64_t size = customers.size(); // Number of customers.

    while (target < size)
    {
        c.username = "user_" + to_string(target); // Create a dummy username.
        customers.at(target) = c; // Insert the user in the customers list.

        h_customers[target] = new char[c.username.length() + 1]; // Aggiunto 1 per il terminatore null.
        strcpy(h_customers[target], c.username.c_str()); // Copy the username in the usernames array.

        target += THREAD_NUMBER_CPU; // Go on.
    }
}

void threadCodeBuildTable(vector<strutturaCustomer> &customers, uint16_t *hashes, vector<vector<strutturaCustomer>> &ret, uint8_t id)
{
    uint64_t target = id; // Starting point of this CPU-Thread.
    uint16_t index; // Contains the value of an hash.
    while (target < NUMBER_OF_CUSTOMERS)
    {
        index = hashes[target]; // Take the hash of the customer.
        mutexVector.mutexes[index].lock(); // Lock on the row.
        ret.at(index).push_back(customers[target]); // Insert the customer.
        mutexVector.mutexes[index].unlock(); // Unlock the row.
        target += THREAD_NUMBER_CPU; // Go on.
    }
}

int main()
{
    uint64_t i = 0; // 64-bit iterator variable.
    uint64_t count = 0; // 64-bit counter variable, used to check the final result.
    vector<strutturaCustomer> customers (NUMBER_OF_CUSTOMERS);  // The list of the customers.
    vector<vector<strutturaCustomer>> ret(HASH_FUNCTION_SIZE); // The final hashing table.

    uint16_t *hashes = new uint16_t[NUMBER_OF_CUSTOMERS]; // Hashes vector.
    char **h_customers = new char *[NUMBER_OF_CUSTOMERS]; // Usernames's vector.

    cudaEvent_t tic, toc; // Variables for compute the execution time of the CUDA kernel.
    float elapsed = 0.0f; // Variable for compute the execution time.

    vector<thread> threadMixer (THREAD_NUMBER_CPU - 1); // Vector of the threads descriptors.
    uint8_t ithread = 0; // 8-bit iterator variable.

   // decltype(std::chrono::steady_clock::now()) start_steady, end_steady; // The definition of the used timer variables.

    (cudaEventCreate(&tic));
    (cudaEventCreate(&toc));

    for (ithread = 0; ithread < THREAD_NUMBER_CPU - 1; ithread++) { // For each started thread...
        thread thread_i(
            threadCodeInitialize, // The thread function.
            ref(customers), // The customers array.
            ref(h_customers), // The usernames array.
            ithread // The thread's id.
        );
        threadMixer.at(ithread) = move(thread_i); // Add the thread descriptor to the thread descriptor vector.
    }

    // The main thread too contribute to the generation of the data stucture.
    threadCodeInitialize(
        ref(customers), // The customers array.
        ref(h_customers), // The usernames array.
        THREAD_NUMBER_CPU - 1 // The thread's id.
    );

    // Now the father wait for all the started threads to finish their execution.
    for (ithread = 0; ithread < THREAD_NUMBER_CPU - 1; ithread++) {
        threadMixer[ithread].join(); // Join the ith thread.
    }

   // start_steady = std::chrono::steady_clock::now(); // Start measuring the execution time of the main process.
    const uint64_t numCustomers = NUMBER_OF_CUSTOMERS;
    cudaMemcpyToSymbol(d_size, &numCustomers, sizeof(uint64_t));


    // Allocazione overflow indexes in GPU.
    uint16_t *d_hashes = 0;
    (cudaMalloc((void **)&d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t))); // Allocazione della memoria sulla GPU per h_overflowIndexes.

    // Allocazione customers in GPU.
    char **d_customers = 0; // Creiamo la tabella di hash nella GPU.
    size_t size_str = 0; // Length of the username.
    char *d_username = 0;
    (cudaMalloc((void **)&d_customers, NUMBER_OF_CUSTOMERS * sizeof(char *)));
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        size_str = (customers[i].username.length() + 1) * sizeof(char); // salvo la lunghezza della stringa corrente.
        (cudaMalloc((void **)&d_username, size_str));  // Copia del nome utente dalla CPU alla GPU.
        (cudaMemcpy(d_username, h_customers[i], size_str, cudaMemcpyHostToDevice)); // Aggiornamento del puntatore del nome utente nella struttura dati sul device.
        (cudaMemcpy(&(d_customers[i]), &d_username, sizeof(char *), cudaMemcpyHostToDevice));
    }


    (cudaEventRecord(tic, 0));
    processCustomers KERNEL_ARGS2(BLOCKS_NUMBER, THREAD_NUMBER_GPU) (d_customers, d_hashes);
    //processCustomers KERNEL_ARGS2((NUMBER_OF_CUSTOMERS/THREAD_NUMBER_GPU) + 1, THREAD_NUMBER_GPU) (d_customers, NUMBER_OF_CUSTOMERS, d_hashes);
    (cudaEventRecord(toc, 0));

    (cudaDeviceSynchronize()); // Sincronizza la GPU per assicurarsi che il kernel sia stato completato.

    (cudaEventSynchronize(toc)); // Synchronize the event.

    (cudaEventElapsedTime(&elapsed, tic, toc)); // Compute the elapsed time.

    (cudaMemcpy(hashes, d_hashes, NUMBER_OF_CUSTOMERS * sizeof(uint16_t), cudaMemcpyDeviceToHost)); // Copia dei risultati dalla GPU alla CPU.
    
    /*
    cout << "Risultati copiati in memoria!" << endl;

    for(i = 0 ; i < NUMBER_OF_CUSTOMERS ; i++){
        cout << hashes[i] << endl;
    }
    */
 

   for (ithread = 0; ithread < THREAD_NUMBER_CPU - 1; ithread++) { // For each started thread...
       thread thread_i(
           threadCodeBuildTable, // The thread function.
           ref(customers), // The customers array.
           ref(hashes), // The hashes array.
           ref(ret), // The hash table.
           ithread // The thread's id.
       );
       threadMixer.at(ithread) = move(thread_i); // Add the thread descriptor to the thread descriptor vector.
   }

   // The main thread too contribute to the generation of the data stucture.
   threadCodeBuildTable(
       ref(customers), // The customers array.
       ref(hashes),  // The hashes array.
       ref(ret),  // The hash table.
       THREAD_NUMBER_CPU - 1// The thread's id.
   );

   // Now the father wait for all the started threads to finish their execution.
   for (ithread = 0; ithread < THREAD_NUMBER_CPU - 1; ithread++) {
       threadMixer[ithread].join(); // Join the ith thread.
   }
   
   // end_steady = std::chrono::steady_clock::now();                                      // Measure the execution time of the main process when all the threads are ended.
   // std::chrono::duration<double> elapsed_seconds_high_res = end_steady - start_steady; // Compute the execution time.
    //const double time = elapsed_seconds_high_res.count();                               // Return the total execution time.

    if(PRINT_CHECKS){
        for (i = 0; i < HASH_FUNCTION_SIZE; i++)
        {
            count += ret[i].size(); // Compute how many struct has each row of the table.
        }

        if (count == NUMBER_OF_CUSTOMERS)   
        // If the table has every struct, then the result is correct.
        {
            cout << "SUCCESS [" << count <<"] - The hash table has been successfully built!" << endl;
        }
        else
        {
             cout << "ERROR [" << count <<"] - The hash table has not been correctly built!" << endl;
        }
    }
  
    // Deallocation
    for (i = 0; i < NUMBER_OF_CUSTOMERS; ++i)
    {
        // Deallocazione di ogni singolo username dalla GPU.
        cudaMemcpy(&d_username, &d_customers[i], sizeof(char *), cudaMemcpyDeviceToHost);
        cudaFree(d_username);
    }

    (cudaFree(d_customers)); // Deallocazione della memoria sulla GPU per d_customers.
    (cudaFree(d_hashes)); // Deallocazione della memoria sulla GPU per d_hashes.

    // Rilascio della memoria CPU allocata
    for (i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
       delete[] h_customers[i];
    }
    delete[] h_customers;
    delete[] hashes;

    // Free the two events tic and toc
    (cudaEventDestroy(tic));
    (cudaEventDestroy(toc));
    
    if (PRINT_CHECKS)
    {
        cout << "-----------------------------------------" << endl;
        cout << "Kernel execution time: " << elapsed << " ms" << endl;
        cout << "Total execution time : " << time << " s" << endl;
        cout << "-----------------------------------------" << endl;
    }

    if (SAMPLE_FILE_PRINT)
    {
        // elapsed must be float, but the function wants double
        printToFile(static_cast<double>(elapsed), "kernel.csv"); // Print the sample in the '.csv' file.
        insertNewLine("kernel.csv");
        //printToFile(time, "total.csv"); // Print the sample in the '.csv' file.
        //insertNewLine("total.csv");
    }

    return 0;
}
