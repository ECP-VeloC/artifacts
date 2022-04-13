#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "veloc.h"
#include <mpi.h>
#include <assert.h>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
using namespace std;
#define SLEEP_TIME 1

__global__ void init_vals(int *d, size_t s, int v) {
    unsigned int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (thread_id < s) {
        d[thread_id] = v+thread_id;
        thread_id += stride;
    }
}

__global__ void verify_vals(int *d, size_t s, int v) {
    unsigned int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (thread_id < s) {
        if (d[thread_id] != v+thread_id && thread_id < 10) {
            printf("--------------- Mismatch in value at index %d: expected: %d, got %d", thread_id, v+thread_id, d[thread_id]);
        }
        thread_id += stride;
    }
}

int main(int argc, char *argv[]) {
    cout << "Test script for checkpoint-restart using VELOC!" << endl;
    int rank, world_size, mpi_supp_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_supp_provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int gpu_id = rank;
    cudaSetDevice(gpu_id);
    if (VELOC_Init(MPI_COMM_WORLD, argv[1]) != VELOC_SUCCESS)
    {
        cout << "Error initializing VELOC! Aborting... " << endl;
        exit(2);
    }
    int nshots = 2;
    int num_snapshots = 800;
    size_t total_size = 0, max_ele;
    unsigned int num_elements;
    vector<vector<size_t>> snapshot_sizes (nshots, vector<size_t>(num_snapshots));
    // Initialize random sizes
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(1<<18, 1<<20); // Ckpt sizes between 1 and 4 MB (int: 4x)
    for(int ishot = 0; ishot < nshots; ishot++) {
        for(int i = 0; i<num_snapshots; i++) {
            snapshot_sizes[ishot][i] = distr(gen);
            num_elements = snapshot_sizes[ishot][i];
            total_size += num_elements;
            if (num_elements > max_ele)
                max_ele = num_elements;
        }
    }
    cout << "Total size of all checkpoints (in MB): " << (total_size*sizeof(int) >> 20) << endl;
    int *da;
    cudaMalloc((void**)&da, max_ele*sizeof(int));
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    for(int ishot = 0; ishot < nshots; ishot++) {
        std::string ckpt_name = std::string("gpu_enabled_veloc_"+std::to_string(ishot));
        cout << "[Rank " << rank  << "] Processing Shot " << ishot << " ckpt name " << ckpt_name << endl; cout.flush();

        // PREFETCH hints
        for(int i=num_snapshots-1; i>=0; i--) 
            assert(VELOC_Prefetch_enqueue(ckpt_name.c_str(), i) == VELOC_SUCCESS);

        // CHECKPOINT
        for(int i=0; i<num_snapshots; i++) {
            cout << "[Rank " << rank  << "] Checkpoint version " << i << " size (bytes): " << snapshot_sizes[ishot][i]*sizeof(int) << endl;
            init_vals<<<1024, 1024>>>(da, snapshot_sizes[ishot][i], ishot*num_snapshots+i);
            assert(VELOC_Mem_protect(0, da, snapshot_sizes[ishot][i], sizeof(int), DEFAULT) == VELOC_SUCCESS);
            assert(VELOC_Checkpoint(ckpt_name.c_str(), i) == VELOC_SUCCESS);
            VELOC_Mem_unprotect(0);
        }
        cout << "[Rank " << rank  << "] All checkpoints written to GPU cache on rank : " << rank << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        assert(VELOC_Prefetch_start() == VELOC_SUCCESS);

        MPI_Barrier(MPI_COMM_WORLD);

        // RESTART AND RELEASE
        for(int i=num_snapshots-1; i>=0; i--) {

            size_t mem_size = VELOC_Recover_ckpt_size(ckpt_name.c_str(), i, 0);
            cout << "[Rank " << rank << " ]  Restarting " << i << " of size " << mem_size << endl;
            num_elements = mem_size/sizeof(int);
            assert(num_elements == snapshot_sizes[ishot][i]);
            assert(VELOC_Mem_protect(0, da, num_elements, sizeof(int), DEFAULT) == VELOC_SUCCESS);
            assert(VELOC_Restart(ckpt_name.c_str(), i) == VELOC_SUCCESS);
            verify_vals<<<1024, 1024>>>(da, num_elements, ishot*num_snapshots+i);
            VELOC_Mem_unprotect(0);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    cudaFree(da);    
    VELOC_Finalize(0);
    MPI_Finalize();
    return 0;
}