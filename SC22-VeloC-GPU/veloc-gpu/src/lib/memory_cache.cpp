#include "include/veloc.hpp"
#include "common/file_util.hpp"
#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include "memory_cache.hpp"
#include <limits>
#include <thread>
#include "client.hpp"
// #define __DEBUG
#include "common/debug.hpp"

memory_cache_t::memory_cache_t(int d){
    device_type = d;
    total_memory = 0;
}

void memory_cache_t::allocate_mem() {
    if (device_type == GPU_MEM) {
        checkCuda(cudaSetDevice(gpu_id));
        checkCuda(cudaMalloc(&start_ptr, total_memory));
    } else {
        checkCuda(cudaMallocHost(&start_ptr, total_memory));
    }
    std::unique_lock<std::mutex> malloc_lock(malloc_mutex);
    max_allocated = total_memory;
    malloc_lock.unlock();
    malloc_cv.notify_one();
}

memory_cache_t::~memory_cache_t() {
    mem_map.clear();
    is_active = false;
    dev_cv.notify_all();
    if (device_type == GPU_MEM) {
        checkCuda(cudaFree(start_ptr));
    } else { 
        checkCuda(cudaFreeHost(start_ptr));
    }
}

void memory_cache_t::init(size_t memory, int g, bool s, bool e) {
    mem_map.clear();
    mem_order.clear();
    mem_starts.clear();
    prefetch_map.clear();
    free_starts.clear();
    free_ends.clear();
    free_sizes.clear();
    p_scores.clear();
    s_scores.clear();
    total_memory = memory;
    score_eviction = s;
    gpu_id = g;
    premature_eviction = e;
    insert_free(total_memory, total_memory);
    if (max_allocated == 0) {
        malloc_thread = std::thread([&] { allocate_mem(); });
        malloc_thread.detach();
    }
}

char* memory_cache_t::insert_entry(mem_region_t *m) {
    m->end_offset = m->start_offset+m->sz;
    std::unique_lock<std::mutex> malloc_lock(malloc_mutex);
    while (m->end_offset > max_allocated)
        malloc_cv.wait(malloc_lock);

    malloc_lock.unlock();
    if(m->end_offset > total_memory) {
        ERROR("End offset: " << m->end_offset << " of checkpoint version " << m->ckpt_version << " is out of cache boundary on " << (device_type == GPU_MEM ? "GPU":"Host" ));
        return 0;
    }

    m->ptr = start_ptr + m->start_offset;
    m->release_at = last_trf_time;
    if (m->sz > max_ckpt_size)
        max_ckpt_size = m->sz;
    std::unique_lock<std::mutex> lock(dev_mutex);
    mem_map[m->uid] = m;
    mem_order.emplace_back(m->uid);
    mem_starts[m->start_offset] = m->uid;
    lock.unlock();
    dev_cv.notify_all();
    last_trf_time++;
    return (char*)m->ptr;
}

void memory_cache_t::insert_free(size_t size, size_t end_offset) {
    // Check if there is an adjoining free region
    size_t start_offset = end_offset-size;
    if(end_offset > total_memory) {
        ERROR("End offset: " << end_offset << " cannot be a free region");
        return;
    }
    if (start_offset > end_offset) {
        ERROR("Start offset: " << start_offset << " cannot be grater than end_offset " << end_offset);
        return;
    }
    auto ite = free_ends.find(start_offset);
    if(ite!=free_ends.end()) {
        size += ite->second->first;
        remove_free(ite);
    }
    ite = free_starts.find(end_offset);
    if(ite != free_starts.end()) {
        size += ite->second->first;
        end_offset += ite->second->first;
        remove_free(ite, true);
    }
    std::unique_lock<std::mutex> lock(free_mutex);
    auto it = free_sizes.emplace(std::make_pair(size, end_offset));
    free_ends.emplace(std::make_pair(end_offset, it));
    free_starts.emplace(std::make_pair(start_offset, it));
}

size_t memory_cache_t::remove_free(offset_t::iterator it, bool is_start) {
    if (is_start && it == free_starts.end())
        return 0;
    else if (it == free_ends.end())
        return 0;
    size_t size = it->second->first;
    std::unique_lock<std::mutex> lock(free_mutex);
    free_sizes.erase(it->second);
    if(is_start) {
        free_ends.erase(it->first+size);
        free_starts.erase(it);
    } else {
        free_starts.erase(it->first-size);
        free_ends.erase(it);
    }
    return size;
}

char* memory_cache_t::get_mem_offset(mem_region_t *m) {  
    m->trf_status = TRF_IN_PROGRESS;
    if(mem_map.find(m->uid) != mem_map.end()) 
        init(total_memory, gpu_id, score_eviction, premature_eviction);
    // Check if space is available on the GPU cache
    auto it = free_sizes.upper_bound(m->sz-1);
    if (it != free_sizes.end()) {
        size_t end_offset = it->second;
        size_t sz = it->first;
        m->start_offset = end_offset-sz;
        size_t rem_space = sz-m->sz;
        remove_free(free_ends.find(end_offset));
        if(rem_space > 0) 
            insert_free(rem_space, end_offset);
        return insert_entry(m);
    }
    if(!score_eviction)
        return _fifo(m);
    return _score_based(m);
}

char* memory_cache_t::_fifo(mem_region_t *m) {
    DBG("In FIFO for " << m->ckpt_version);
    size_t space=0, start = 0, end_offset = 0;
    mem_region_t *e; offset_t::iterator ite;
    // FIFO: removing oldest
    while(space < m->sz) {
        e = mem_map[mem_order.front()];
        std::unique_lock<std::mutex> lock(dev_mutex);
        while(
            (e->trf_status == TRF_IN_PROGRESS || 
            e->pf_status == PREFETCH_STARTED ||
            e->pf_status == PREFETCH_COMPLETED)
            && is_active) {
                DBG("Waiting in FIFO for " << e->ckpt_version << " uid " << e->uid << " pf status " << e->pf_status << " trf status " << e->trf_status << " to accomodate " << m->ckpt_version << " for device " << (device_type == GPU_MEM ? "GPU":"Host" ) );
                if(e->pf_status == PREFETCH_CONSUMED && premature_eviction)
                    break;
                dev_cv.wait(lock);
        }
        lock.unlock();
        dev_cv.notify_all();
        if (!is_active)
            return 0;
        end_offset = start + space;
        if(space == 0 || (end_offset != e->start_offset)) {
            if (space != 0) {
                insert_free(space, end_offset);
                space = 0;
            }
            start = e->start_offset;
            size_t free_size = remove_free(free_ends.find(start));
            space += free_size;
            start -= free_size;
        }
        space += e->sz;
        space += remove_free(free_starts.find(e->end_offset), true);
        lock.lock();
        mem_starts.erase(e->start_offset);
        mem_map.erase(e->uid);
        mem_order.pop_front();
        lock.unlock();
        dev_cv.notify_all();
    }
    if(space-m->sz) 
        insert_free(space-m->sz, start+space);
    m->start_offset = start;
    return insert_entry(m);
}

size_t memory_cache_t::_find_free_neighbor(size_t offset, bool is_start) {
    if (is_start) {
        auto it = free_starts.find(offset);
        if (it != free_starts.end())
            return it->second->first;
        return 0;
    } else {
        auto it = free_ends.find(offset);
        if (it != free_ends.end())
            return it->second->first;
        return 0;
    }
}

char* memory_cache_t::_score_based(mem_region_t *m) {
    DBG("In score-based for " << m->ckpt_version << " for device " << (device_type == GPU_MEM ? "GPU":"Host" ) );
    rerun_algo = 0;
    mem_region_t *e;
    size_t space = 0, start, prev_free = 0;
    double p_score = 0, s_score = 0;
    double min_p_score = std::numeric_limits<double>::max(), min_s_score = std::numeric_limits<double>::max(); 
    mem_start_t::iterator i=mem_starts.begin(), j=mem_starts.begin(), prev, reg_start, reg_end;
    while(i != mem_starts.end() && j != mem_starts.end()) {
        if (i != mem_starts.begin()) {
            if (prev == j) {
                p_score = s_score = space = 0;
                ++j;
            } else {
                p_score -= p_scores[prev->second];
                s_score -= s_scores[prev->second];
                space -= sizes[prev->second] - prev_free;
            }
        }
        prev = i;
        e = mem_map[i->second];
        prev_free = _find_free_neighbor(e->start_offset, true);
        space += prev_free;
        while(j!=mem_starts.end() && space < m->sz) {
            unsigned long int uid = j->second;
            e = mem_map[uid];
            p_scores[uid] = e->trf_status == TRF_COMPLETED ? 0: e->release_at;
            if (s_scores.find(uid) == s_scores.end() || s_scores[uid] >= 0){
                s_scores[uid] = 0;
                auto it = prefetch_map.find(uid); 
                if (it != prefetch_map.end() && mem_map[uid]->pf_status != PREFETCH_CONSUMED) {
                    s_scores[uid] = (1.1-(double)it->second/prefetch_map.size());
                }
            }
            sizes[uid] = e->sz;
            sizes[uid] += _find_free_neighbor(e->start_offset+e->sz, false); // Find if there is a free block after this mem region ends. O(1)  
            p_score += p_scores[uid];
            s_score += s_scores[uid];
            space += sizes[uid];
            ++j;
        }
        if ( space >= m->sz && (
                (p_score < min_p_score) || 
                (p_score == min_p_score && s_score < min_s_score)
            )) {
            min_p_score = p_score;
            min_s_score = s_score;
            reg_start = i;
            reg_end = j;
        }
        ++i;
    }
    
    start = mem_map[reg_start->second]->start_offset;
    space = remove_free(free_ends.find(start), false);
    start -= space;
    // Remove the cached-regions with lowest scores
    for(i=reg_start; i!=reg_end && is_active; ++i) {
        e = mem_map[i->second];        
        space += e->sz;
        space += remove_free(free_starts.find(e->start_offset+e->sz), true);
        std::unique_lock<std::mutex> lock(dev_mutex);
        while(
            (e->trf_status == TRF_IN_PROGRESS || 
            e->pf_status == PREFETCH_STARTED ||
            e->pf_status == PREFETCH_COMPLETED)
            && is_active) {
                if(e->pf_status == PREFETCH_CONSUMED && premature_eviction)
                    break;
                if(rerun_algo > 0) {
                    DBG("Rerun algo: " << rerun_algo << " to accomodate " << m->ckpt_version << " for device " << (device_type == GPU_MEM ? "GPU":"Host" ));
                    rerun_algo--;
                    lock.unlock();
                    return _score_based(m);
                }
                dev_cv.wait(lock);
        }
        if (!is_active)
            return 0;
        
    }
    std::unique_lock<std::mutex> lock(dev_mutex);
    for(i=reg_start; i!=reg_end && is_active; ++i) {
        mem_map.erase(i->second);
    }
    mem_starts.erase(reg_start, reg_end);
    lock.unlock();
    if(space > m->sz)
        insert_free(space-m->sz, start+space);
    m->start_offset = start;
    return insert_entry(m);
}

void memory_cache_t::add_prefetch(unsigned long long int uid) {
    std::unique_lock<std::mutex> lock(dev_mutex);
    if (prefetch_map.find(uid) != prefetch_map.end())
        prefetch_map.clear();
    prefetch_map[uid] = prefetch_map.size()+1;
    lock.unlock();
    dev_cv.notify_all();
}

void memory_cache_t::del_prefetch(unsigned long long int uid) {
    std::unique_lock<std::mutex> lock(dev_mutex);
    s_scores[uid] = -(double)prefetch_map.size();
    prefetch_map.erase(uid);    
    lock.unlock();
    dev_cv.notify_all();
    set_pf(uid, PREFETCH_CONSUMED);
}

int memory_cache_t::set_trf(unsigned long long int uid, int trf_status) {
    std::unique_lock<std::mutex> lock(dev_mutex);
    if(mem_map.find(uid) == mem_map.end()) {
        dev_cv.notify_all();
        return VELOC_FAILURE;
    }    
    mem_region_t* e = mem_map[uid];
    e->trf_status = trf_status;
    lock.unlock();
    dev_cv.notify_all();
    DBG("Marking trf status of " << e->ckpt_version << " uid " << e->uid << " as " << trf_status << " for device " << (device_type == GPU_MEM ? "GPU":"Host" ) );
    return VELOC_SUCCESS;
}

int memory_cache_t::set_pf(unsigned long long int uid, int pf_status) {
    std::unique_lock<std::mutex> lock(dev_mutex);
    if(mem_map.find(uid) == mem_map.end()) {
        dev_cv.notify_all();
        return VELOC_FAILURE;
    }
    mem_region_t* e = mem_map[uid];
    e->pf_status = pf_status;
    // Uncomment the below while loop to ensure consistency of data.
    // while (e->trf_status != TRF_COMPLETED && pf_status == PREFETCH_COMPLETED)
    //     dev_cv.wait(lock);    
    if(pf_status != PREFETCH_CONSUMED) {
        e->release_at = last_trf_time;
        last_trf_time++;
    } else {
        e->release_at = 0;
        rerun_algo++;
    }
    lock.unlock();
    dev_cv.notify_all();
    return VELOC_SUCCESS;
}

void memory_cache_t::print_mem() {
    DBG("=== PRINTING ALLOCS === total size: " << total_memory << " for device " << (device_type == GPU_MEM ? "GPU":"Host" ) );
    std::unique_lock<std::mutex> lock(dev_mutex);
    for (auto i = mem_starts.begin(); i!=mem_starts.end(); i++) { 
        mem_region_t *e = mem_map[i->second];
        DBG("ID " << e->ckpt_version << " trf_status: " << e->trf_status << " pf_status " << e->pf_status << " release at: " << e->release_at);
    }
    DBG("===== Printing frees ===== on " << (device_type == GPU_MEM ? "GPU":"Host" ));
    for(auto it = free_sizes.begin(); it != free_sizes.end(); ++it) {
        DBG("Offset: " << it->second-it->first << " : " << it->second << " = " << it->first);
    }
    lock.unlock();
    dev_cv.notify_all();
}

mem_region_t* memory_cache_t::search(unsigned long long int uid, int pf_status) {
    std::unique_lock<std::mutex> lock(dev_mutex);
    if(mem_map.find(uid) == mem_map.end()) {
        lock.unlock();
        dev_cv.notify_all();
        return nullptr;
    }
    auto e = mem_map[uid];
    lock.unlock();
    dev_cv.notify_all();
    if(pf_status != NO_PREFETCH) 
        set_pf(uid, pf_status);
    return e;
}

bool memory_cache_t::set_rest_rel(unsigned long int uid) {
    std::unique_lock<std::mutex> lock(dev_mutex);
    for(auto it=mem_map.cbegin(); it!=mem_map.cend();++it) {
        if(it->first != uid)
            it->second->pf_status = NO_PREFETCH;
    }
    lock.unlock();
    dev_cv.notify_all();
    return VELOC_SUCCESS;
}