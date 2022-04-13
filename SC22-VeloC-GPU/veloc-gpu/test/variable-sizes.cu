// Compile:
// nvcc -std=c++17 -I//openmpi-4.0.5/include -L//openmpi-4.0.5/lib/ -lmpi -I $HOME/veloc-build/include/ -I $HOME/veloc-build/bin -L$HOME/veloc-build/lib -lveloc-client $HOME/veloc-gpu-tests/uniform-sizes.cu

// Execute:
// time mpirun -np $nprocs a.out /home/am6429/veloc-gpu-tests/fifo-eviction.cfg $ckpt_int_ms $size $num_ckpts $pattern $num_prefetches
// time mpirun -np 1 a.out /home/am6429/veloc-gpu-tests/fifo-eviction.cfg 10 26 800 2 1

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
#include <boost/algorithm/string.hpp>
using namespace std;
using namespace std::chrono;
#define MAX_LEN 1024

void deleteDirectoryContents(string config_file) {
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
    for (auto entry : std::filesystem::directory_iterator(scratch_dir)) {
        std::filesystem::remove_all(entry.path());
    }
    for (auto entry : std::filesystem::directory_iterator(persistent_dir)) 
        std::filesystem::remove_all(entry.path());
}


void read_record(string traces, vector<vector<size_t>> & res) {
	// File pointer
	ifstream fin(traces);
	if(!fin.is_open()) 
        throw std::runtime_error("Could not open file");
	vector<size_t> row;
	string line, word;
	while (fin.good()) {
		row.clear();
		getline(fin, line);
		stringstream s(line);
		while (getline(s, word, ',')) {
			row.push_back(std::stoul(word));
		}
        if(row.size() > 0) 
            res.push_back(row);
	}
}


int main(int argc, char *argv[]) {
    cout << "Test script for checkpoint-restart using VELOC!" << endl;

    int sleep_time = 15;
    int prefetch_order = 0;
    int num_prefetches = 0;
    int num_checkpoints = 0;
    int nshots = 1;
    string traces = "";
    if(argc < 2) {
        cout << "Need a .csv file for traces";
        return -1;
    }
    if (argc > 2) {
        traces = argv[2];
    }
    if (argc > 3) {
        nshots = stoi(argv[3]);
    }
    if (argc > 4) {
        sleep_time = stoi(argv[4]);
    }
    if (argc > 5) {
        prefetch_order = stoi(argv[5]);
        // 0 = Same as checkpoint
        // 1 = Reverse
        // 2 = Random
    }
    if (argc > 6) {
        num_prefetches = stoi(argv[6]);
        // 0 = No prefetching approach
        // 1 = Single prefetch
        // 2 = Multiple prefetches
    }

    vector<vector<size_t>> snapshots;
    read_record(traces, snapshots);
    num_checkpoints = snapshots.size();

    int rank, world_size, mpi_supp_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_supp_provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int total_gpus = 0;
    cudaGetDeviceCount(&total_gpus);
    int gpu_id = rank%total_gpus;
    cout << "Rank " << rank << " running on GPU " << gpu_id << endl;
    cudaSetDevice(gpu_id);
    if (VELOC_Init(MPI_COMM_WORLD, argv[1]) != VELOC_SUCCESS)
    {
        cout << "Error initializing VELOC! Aborting... " << endl;
        exit(2);
    }

    size_t total_size = 0, size = 0;
    for(int i=0; i<num_checkpoints; i++) {
        total_size += snapshots[i][gpu_id];
        if(snapshots[i][gpu_id] > size)
            size = snapshots[i][gpu_id];
    }

    // Generate prefetch order
    vector<int> pf_order(num_checkpoints);
    std::iota (pf_order.begin(), pf_order.end(), 0);
    if(prefetch_order == 1) {
        int n = num_checkpoints-1;
        std::generate(pf_order.begin(), pf_order.end(), [&n]{ return n--;});
    } else if (prefetch_order == 2) {
        int random_number = 10;
        std::mt19937 g(random_number);
        std::shuffle(pf_order.begin(), pf_order.end(), g);
    }

    if(rank == 0) {
        cout << "Total number of checkpoints: " << num_checkpoints << endl;
        cout << "Total number of shots: " << nshots << endl;
        cout << "Checkpoint interval (in ms): " << sleep_time << endl;
        cout << "Prefetch order " << prefetch_order << endl;
        cout << "Num prefetches " << num_prefetches << endl;
    } 
    cout << "[Rank " << rank << "] Total size of all checkpoints (in MB): " << (nshots*total_size*sizeof(char) >> 20) << endl;
    cout << "[Rank " << rank << "] Max size of a checkpoints (in MB): " << (size*sizeof(char) >> 20) << endl;
    int *da;
    cudaMalloc((void**)&da, size*sizeof(char));

    MPI_Barrier(MPI_COMM_WORLD);
    
    vector<unsigned int> ckpt_times(num_checkpoints);
    vector<unsigned int> restore_times(num_checkpoints);
    vector<unsigned int> next_prefetched(num_checkpoints, 0);
    unsigned int shot_time;
    high_resolution_clock::time_point st, en,  shot_time_st, shot_time_en;
    char c[MAX_LEN] = {};
    for(int ishot = 0; ishot < nshots; ishot++) {
        if(rank == 0) {
            cout << "Shot " << ishot << endl; 
            cout.flush();
        }
        shot_time_st = high_resolution_clock::now();
        std::string s = std::string("gpu_enabled_veloc_"+std::to_string(ishot));
        const char * ckpt_name = s.c_str();
        // PREFETCH hints
        if (num_prefetches == 2) {
            for(auto &i: pf_order)
                VELOC_Prefetch_enqueue(ckpt_name, i);
        }
        // CHECKPOINT
        for(int i=0; i<num_checkpoints; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            std::cout << std::flush; // flushed!
            assert(VELOC_Mem_protect(0, da, snapshots[i][gpu_id], sizeof(char), DEFAULT) == VELOC_SUCCESS);
            st = high_resolution_clock::now();
            assert(VELOC_Checkpoint(ckpt_name, i) == VELOC_SUCCESS);
            en = high_resolution_clock::now();
            ckpt_times[i] = duration_cast<nanoseconds>(en - st).count();
            VELOC_Mem_unprotect(0);
        }      
        
        assert(VELOC_Checkpoint_wait() == VELOC_SUCCESS);
        cout << "[Rank " << rank << "] Checkpoints written to scratch " << endl;
        
        if(num_prefetches == 2)
            assert(VELOC_Prefetch_start() == VELOC_SUCCESS);
        
        // RESTART AND RELEASE
        for(int x=0; x<pf_order.size(); x++) {
            int i= pf_order[x];
            if (num_prefetches == 1) {
                VELOC_Prefetch_enqueue(ckpt_name, i);
                assert(VELOC_Prefetch_start() == VELOC_SUCCESS);
            } 
            cout << "[Rank " << rank << "] Snap " << ishot << ", Restarting checkpoint " << i << endl;
        
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            size_t mem_size = VELOC_Recover_ckpt_size(ckpt_name, i, 0);
            assert(mem_size == snapshots[i][gpu_id]);
            assert(VELOC_Mem_protect(0, da, snapshots[i][gpu_id], sizeof(char), DEFAULT) == VELOC_SUCCESS);
            st = high_resolution_clock::now();
            assert(VELOC_Restart(ckpt_name, i) == VELOC_SUCCESS);
            en = high_resolution_clock::now();
            restore_times[i] = duration_cast<nanoseconds>(en - st).count();
            VELOC_Mem_unprotect(0);
            
            for(int y=x+1; y<pf_order.size(); y++) {
                int j = pf_order[y];
                if (VELOC_Next_prefetched(ckpt_name, j, 0) == false)
                    break;
                next_prefetched[x]++;
            }
        }
        shot_time_st = high_resolution_clock::now();
        shot_time = duration_cast<nanoseconds>(shot_time_en - shot_time_st).count();
        VELOC_Get_trf_metrics(c);
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

        if(rank == 0) {
            for(auto &i: pf_order)
                cout << i << ", ";
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
            deleteDirectoryContents(argv[1]);
        }
    }
    cudaFree(da);    
    VELOC_Finalize(0);
    MPI_Finalize();    
    return 0;
}