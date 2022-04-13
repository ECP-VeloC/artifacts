#ifndef __CLIENT_HPP
#define __CLIENT_HPP

#include "include/veloc.hpp"

#include "common/config.hpp"
#include "common/command.hpp"
#include "common/comm_queue.hpp"
#include "modules/module_manager.hpp"

#include <unordered_map>
#include <map>
#include <deque>
#include <queue> 
#include "memory_cache.hpp"

class client_impl_t : public veloc::client_t {
    config_t cfg;
    MPI_Comm comm, local = MPI_COMM_NULL, backends = MPI_COMM_NULL;
    bool collective, ec_active;
    int rank;
    command_t current_ckpt;
    command_t pf_cmd;
    bool checkpoint_in_progress = false;
    std::map<int, size_t> region_info;
    std::map<int, size_t> region_info_pf;
    std::map<int, size_t> region_f_offset;
    size_t header_size = 0;
    size_t header_size_pf = 0;
    comm_client_t<command_t> *queue = NULL;
    module_manager_t *modules = NULL;
    typedef std::map<int, mem_region_t> regions_t;
    int gpu_id;
    regions_t mem_regions;
    regions_t ckpt_regions;
    cudaStream_t veloc_stream;
    cudaStream_t veloc_gpu_to_host_stream;
    cudaStream_t veloc_rec_stream;
    std::thread gpu_memcpy_thread;
    std::thread write_to_file_thread;
    std::thread prefetch_thread;
    size_t total_gpu_cache = 0;
    size_t total_host_cache = 0; 
    bool start_prefetching = false;
    std::queue<unsigned long long int> prefetch_order;
    std::queue<unsigned long long int> prefetch_order_restart;
    std::unordered_map<unsigned long long int, mem_region_t *> prefetch_map;
    std::mutex prefetch_mutex;
    std::condition_variable prefetch_cv;
    std::map<int, std::map<int, size_t>> ckpt_meta; // ckpt_version, region_id, size
    std::map<int, int> num_regions; // ckpt_version
    std::mutex ckpt_mutex;
    std::condition_variable ckpt_cv;
    std::queue<mem_region_t*> gpu_write_q;
    std::mutex gpu_write_mutex;
    std::condition_variable gpu_write_cv;
    std::queue<mem_region_t*> file_write_q;
    std::mutex file_write_mutex;
    std::condition_variable file_write_cv;
    memory_cache_t gpu_cache;
    memory_cache_t host_cache;
    bool veloc_client_active = true;
    std::map<std::string, size_t> trf_metrics;

    int run_blocking(const command_t &cmd);
    bool read_header(bool is_pf = false);
    void launch_threaded(MPI_Comm comm, const std::string &cfg_file);

public:
    client_impl_t(unsigned int id, const std::string &cfg_file);
    client_impl_t(MPI_Comm comm, const std::string &cfg_file);

    bool mem_protect(int id, void *ptr, size_t count, size_t base_size, unsigned int flags, release_routine r);
    virtual std::string route_file(const std::string &original);

    virtual bool checkpoint(const std::string &name, int version);
    virtual bool checkpoint_begin(const std::string &name, int version);
    virtual bool checkpoint_mem(int mode, const std::set<int> &ids);
    virtual bool checkpoint_end(bool success);
    virtual bool checkpoint_wait();

    virtual int restart_test(const std::string &name, int version);
    virtual bool restart(const std::string &name, int version);
    virtual bool restart_begin(const std::string &name, int version);
    virtual size_t recover_size(int id);
    virtual bool recover_mem(int mode, const std::set<int> &ids);
    virtual bool restart_end(bool success);
    
    void gpu_to_host_trf();
    void mem_to_file_trf();

    void prefetch_restart();
    size_t recover_size(char * ckpt_name, int version, int region_id);
    int prefetch_enqueue(const char *name, int version, int region_id = 0);
    int prefetch_start();
    bool recover_mem_prefetch(unsigned long long int, int res = 0);
    bool checkpoint_end(command_t current_ckpt);
    bool next_prefetched(const char *name, int version, int region_id = 0);
    void get_trf_metrics(char * c);
    virtual ~client_impl_t();
};

#endif
