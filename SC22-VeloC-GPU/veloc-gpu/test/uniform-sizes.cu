// Script to checkpoint-restore uniform sized data elements from GPU to SSD
// Input parameters:
// 1. Configuration file
// 2. Checkpoint/Restore interval (in ms): Will sleep for this interval instead of performing computations
// 3. Size ($size): Char array of size 2^($size), checkpoint size per region = 2^$size
// 4. Number of checkpoints per shot
// 5. Number of shots
// 6. Prefetch pattern: 0: Ascending order, 1: Reverse order, 2: Random order
// 7. Number of prefetches enqueued:  0: No prefetching, 1: Single prefetch, 2: Multiple prefetches

// Compile:
// nvcc -std=c++17 -I//openmpi-4.0.5/include -L//openmpi-4.0.5/lib/ -lmpi 
// -I $HOME/veloc-build/include/ -I $HOME/veloc-build/bin -L$HOME/veloc-build/lib -lveloc-client $HOME/veloc-gpu-tests/uniform-sizes.cu -o uniform-sizes.out

// Execute:
// time mpirun -np $nprocs uniform-sizes.out /home/am6429/veloc-gpu-tests/fifo-eviction.cfg $ckpt_int_ms $size $num_ckpts $num_shots $pattern $num_prefetches
// E.g. time mpirun -np 1 uniform-sizes.out /home/am6429/veloc-gpu-tests/fifo-eviction.cfg 10 26 800 10 2 1

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
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
using namespace std;
using namespace std::chrono;
#define MAX_LEN 1024

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void deleteDirectoryContents(string config_file, string ckpt_name, int rank, int num_checkpoints) {
    string scratch_dir = "";
    string persistent_dir = "";
    fstream newfile;
    newfile.open(config_file, ios::in);
    if (newfile.is_open()) {
        string temp;
        size_t pos;
        while(getline(newfile, temp)) {
            pos = temp.find("/");
            if ((temp.rfind("scratch=", 0) != string::npos && pos != string::npos) || (temp.rfind("scratch =", 0) != string::npos && pos != string::npos))
                scratch_dir = temp.substr(pos, temp.length()-pos);
            if ((temp.rfind("persistent=", 0) != string::npos && pos != string::npos) || (temp.rfind("persistent =", 0) != string::npos && pos != string::npos))
                persistent_dir = temp.substr(pos, temp.length()-pos);
        }
    }

    for (int i=0; i<num_checkpoints; i++) {
        try {
            string filename = scratch_dir + std::string(ckpt_name) + "-" + std::to_string(rank) + "-" + std::to_string(i) + ".dat";
            std::filesystem::remove(filename);
        }
        catch(const std::filesystem::filesystem_error& err) {
            std::cout << "filesystem error: " << err.what() << endl;
        }
    }
}

int main(int argc, char *argv[]) {
    cout << "Test script for checkpoint-restart using VELOC for uniform checkpoint sizes!" << endl;
    size_t size = (1<<26);
    int num_checkpoints = 64;
    int nshots = 2;
    int sleep_time = 15;
    int prefetch_order = 0;
    int num_prefetches = 0;
    if (argc > 2)
        size = 1<<stoi(argv[2]);
    if (argc > 3)
        nshots = stoi(argv[3]);
    if (argc > 4)
        num_checkpoints = stoi(argv[4]);
    if (argc > 5) 
        sleep_time = stoi(argv[5]);
    if (argc > 6) {
        prefetch_order = stoi(argv[6]);
        // 0 = Same as checkpoint
        // 1 = Reverse
        // 2 = Random
    }
    if (argc > 7) {
        num_prefetches = stoi(argv[7]);
        // 0 = No prefetching approach
        // 1 = Single prefetch
        // 2 = Multiple prefetches
    }

    int rank, world_size, mpi_supp_provided;
    MPI_Comm comm = MPI_COMM_WORLD, app_comm;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_supp_provided);
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &rank);

    MPI_Query_thread(&mpi_supp_provided);
    MPI_Comm_split(comm, mpi_supp_provided == 0 ? 0 : MPI_UNDEFINED, rank, &app_comm);
    cout << "[Rank " << rank << "] MPI support provided " << mpi_supp_provided << endl;

    int total_gpus = 0;
    checkCuda(cudaGetDeviceCount(&total_gpus));
    int gpu_id = rank%total_gpus;
    checkCuda(cudaSetDevice(gpu_id));

    int *da;
    cudaMalloc((void**)&da, size*sizeof(char));
    // START: Helper for generating different prefetch orders, not required for RTM
    vector<int> pf_order(num_checkpoints);
    std::iota (pf_order.begin(), pf_order.end(), 0);
    if(prefetch_order == 1) {
        int n = num_checkpoints-1;
        std::generate(pf_order.begin(), pf_order.end(), [&n]{ return n--;});
    } else if (prefetch_order == 2) {
        int random_number = 101;
        std::mt19937 g(random_number);
        std::shuffle(pf_order.begin(), pf_order.end(), g);
    }
    // END: Helper for generating different prefetch orders, not required for RTM

    if(rank == 0) {
        cout << "Size of each snapshot (in MB) " << (size*sizeof(char) >> 20) << ", total number of snapshots: " << num_checkpoints << endl;
        cout << "Total size of all snapshots (in MB): " << (size*sizeof(char)*num_checkpoints >> 20) << endl;
        cout << "Total number of checkpoints: " << num_checkpoints << endl;
        cout << "Total number of shots: " << nshots << endl;
        cout << "Checkpoint interval (in ms): " << sleep_time << endl;
        cout << "Prefetch order " << prefetch_order << endl;
        cout << "Num prefetches " << num_prefetches << endl;
    }

    if (VELOC_Init(comm, argv[1]) != VELOC_SUCCESS) {
        cout << "Error initializing VELOC! Aborting... " << endl;
        exit(2);
    }
    

    VELOC_Mem_protect(0, da, size, sizeof(char), DEFAULT);
    
    vector<unsigned int> ckpt_times(num_checkpoints);
    vector<unsigned int> restore_times(num_checkpoints);
    vector<unsigned int> next_prefetched(num_checkpoints, 0);
    unsigned int shot_time;
    high_resolution_clock::time_point st, en, shot_time_st, shot_time_en;
    char c[MAX_LEN] = {};
    for(int ishot=0; ishot<nshots; ishot++) {
        if(rank == 0)
            cout << "Shot " << ishot << endl; cout.flush();
        // Every shot should have a unique checkpoint name
        shot_time_st = high_resolution_clock::now();
        string s = std::string("gpu_enabled_veloc_"+std::to_string(ishot));
        const char * ckpt_name = s.c_str();
        // START: Enqueue prefetches
        if (num_prefetches == 2) {
            for(auto &i: pf_order)
                VELOC_Prefetch_enqueue(ckpt_name, i);
        }
        // END: Enqueue prefetches
    
        // START: CHECKPOINT
        for(int i=0; i<num_checkpoints; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time)); // Run kernel
            st = high_resolution_clock::now();
            VELOC_Checkpoint(ckpt_name, i);
            en = high_resolution_clock::now();
            ckpt_times[i] = duration_cast<nanoseconds>(en - st).count();
            // cout << "[Rank " << rank << "] Checkpointing shot " << ishot << " checkpoint version " << i << endl;
        }
        // END: CHECKPOINT
        // We wait for the flushes to complete so that they do not consume bandwidth during restore phase.
        assert(VELOC_Checkpoint_wait() == VELOC_SUCCESS);
        cout << "[Rank " << rank << "] Checkpoints written to scratch " << endl;
        if(num_prefetches == 2) 
            VELOC_Prefetch_start();    

        // START: RESTORE
        for(int x=0; x<pf_order.size(); x++) {
            int i= pf_order[x];
            if (num_prefetches == 1) {
                VELOC_Prefetch_enqueue(ckpt_name, i);
                assert(VELOC_Prefetch_start() == VELOC_SUCCESS);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time)); // Run kernel
            // cout << "[Rank " << rank << "] Restarting shot " << ishot << " checkpoint version " << i << endl;
            st = high_resolution_clock::now();
            VELOC_Restart(ckpt_name, i);
            en = high_resolution_clock::now();
            restore_times[i] = duration_cast<nanoseconds>(en - st).count();
            
            // Check how many of the next prefetches have already been brought to the GPU cache.
            for(int y=x+1; y<pf_order.size(); y++) {
                int j = pf_order[y];
                if (VELOC_Next_prefetched(ckpt_name, j, 0) == false)
                    break;
                next_prefetched[x]++;
            }
        }
        shot_time_st = high_resolution_clock::now();
        shot_time = duration_cast<nanoseconds>(shot_time_en - shot_time_st).count();
        // END: RESTORE
        VELOC_Get_trf_metrics(c);
        deleteDirectoryContents(std::string(argv[1]), s, rank, num_checkpoints);
        // START: Record metrics
        char * trf_m = (char *)malloc(world_size*sizeof(char)*MAX_LEN);
        unsigned int *cbuf = (unsigned int *)malloc(world_size*sizeof(unsigned int)*num_checkpoints);
        unsigned int *rbuf = (unsigned int *)malloc(world_size*sizeof(unsigned int)*num_checkpoints);
        unsigned int *npbuf = (unsigned int *)malloc(world_size*sizeof(unsigned int)*num_checkpoints);
        unsigned int *shot_times = (unsigned int *)malloc(world_size*sizeof(unsigned int));
        MPI_Gather(c, MAX_LEN, MPI_CHAR, trf_m, MAX_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Gather(&ckpt_times[0], num_checkpoints, MPI_UNSIGNED, cbuf, num_checkpoints, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Gather(&restore_times[0], num_checkpoints, MPI_UNSIGNED, rbuf, num_checkpoints, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Gather(&next_prefetched[0], num_checkpoints, MPI_UNSIGNED, npbuf, num_checkpoints, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Gather(&shot_time, 1, MPI_UNSIGNED, shot_times, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        // END: Record metrics

        // START: Report metrics
        if(rank == 0) {
            for(int i=0; i<num_checkpoints; i++) {
                cout << pf_order[i] << ", ";
            }
            cout << endl;
            for(int i=0; i<world_size; i++) {
                for (int j=0; j<num_checkpoints; j++) {
                    cout << npbuf[i*num_checkpoints+j] << ", ";
                }
                cout << endl;
            }
            for(int i=0; i<world_size; i++) {
                for (int j=0; j<num_checkpoints; j++) {
                    cout << cbuf[i*num_checkpoints+j] << ", ";
                }
                cout << endl;
            }
            for(int i=0; i<world_size; i++) {
                for(int j=0; j<num_checkpoints; j++)
                    cout << rbuf[i*num_checkpoints+j] << ", ";
                cout << endl;
            }
            for(int i=0; i<world_size; i++) {
                cout << trf_m+(i*MAX_LEN) << endl;
            }
            for(int i=0; i<world_size; i++) {
                cout << shot_times[i] << ", ";
            }
            cout << endl;
        }
        // END: Report metrics
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    VELOC_Mem_unprotect(0);
    cudaFree(da);    
    VELOC_Finalize(0);
    MPI_Finalize();
    return 0;
}


