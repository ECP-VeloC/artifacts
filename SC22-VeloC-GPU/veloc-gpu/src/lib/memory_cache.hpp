#ifndef __MEMORY_CACHE_HPP
#define __MEMORY_CACHE_HPP

#include "include/veloc.h"
#include "common/file_util.hpp"
#include <stdlib.h>
#include <limits>
#include "mem_region.hpp"
#include <queue>
#include <map>
#include <thread>

class memory_cache_t {
    size_t total_memory;
    int device_type;
    std::mutex dev_mutex;
    std::condition_variable dev_cv;

    std::mutex free_mutex;
    std::condition_variable free_cv;

    mem_map_t mem_map;
    // TODO: Remove this
    std::deque<unsigned long long int> mem_order;
    typedef std::map<size_t, unsigned long int> mem_start_t;
    mem_start_t mem_starts;
    std::unordered_map<unsigned long int, int> prefetch_map;
    unsigned int last_trf_time = 0;
    typedef std::multimap<size_t, size_t> space_t;
    typedef std::unordered_map<size_t, space_t::iterator> offset_t;
    space_t free_sizes;     // <size, end_offset>
    offset_t free_ends;     // <end_offset, iterator to free_size multimap>
    offset_t free_starts;   // <start_offset, iterator_to_free_size_multimap>

    bool score_eviction = false;
    typedef std::unordered_map<unsigned long long int, double> score_t;
    score_t p_scores, s_scores;
    std::unordered_map<unsigned long int, size_t> sizes;
    size_t max_ckpt_size = 0;

    bool is_active = true;
    std::thread malloc_thread;
    size_t max_allocated = 0;
    std::mutex malloc_mutex;
    std::condition_variable malloc_cv;
    char* start_ptr;
    int gpu_id = -1;
    bool premature_eviction = false;
    int rerun_algo = 0;

    public:
    memory_cache_t(int d);
    ~memory_cache_t();
    void init(size_t m, int gpu_id = -1, bool s = false, bool premature_eviction = false);

    void allocate_mem();
    char* insert_entry(mem_region_t *m);
    char* get_mem_offset(mem_region_t *m);
    char* _fifo(mem_region_t *m);
    char* _score_based(mem_region_t *m);
    int set_trf(unsigned long long int uid, int trf_status);
    int set_pf(unsigned long long int uid, int pf_status);
    void print_mem();
    mem_region_t* search(unsigned long long int uid, int pf_status = NO_PREFETCH);
    void add_prefetch(unsigned long long int uid);
    void del_prefetch(unsigned long long int uid);
    void insert_free(size_t size, size_t end_offset);
    size_t remove_free(offset_t::iterator, bool is_start = false);
    bool set_rest_rel(unsigned long int uid);
    size_t _find_free_neighbor(size_t offset, bool is_start);
};
#endif // MEMORY_CACHE_HPP
