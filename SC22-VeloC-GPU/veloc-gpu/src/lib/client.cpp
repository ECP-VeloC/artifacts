#include "client.hpp"
#include "common/file_util.hpp"
#include "backend/work_queue.hpp"

#include <fstream>
#include <stdexcept>
#include <regex>
#include <future>
#include <queue>

#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#include <chrono>

// #define __DEBUG
#include "common/debug.hpp"

static inline bool validate_name(const std::string &name) {
    std::regex e("[a-zA-Z0-9_\\.]+");
    return std::regex_match(name, e);
}

static void launch_backend(const std::string &cfg_file) {
    char *path = getenv("VELOC_BIN");
    std::string command;
    if (path != NULL)
        command = std::string(path) + "/";
    command += "veloc-backend " + cfg_file + " --disable-ec > /dev/null";
    if (system(command.c_str()) != 0)
        FATAL("cannot launch active backend for async mode, error: " << strerror(errno));
}

client_impl_t::client_impl_t(unsigned int id, const std::string &cfg_file) :
    cfg(cfg_file, false), collective(false), gpu_cache(GPU_MEM), host_cache(HOST_MEM), rank(id) {
    if (cfg.is_sync()) {
	modules = new module_manager_t();
	modules->add_default_modules(cfg);
    } else {
        launch_backend(cfg_file);
	queue = new comm_client_t<command_t>(rank);
    }
    ec_active = run_blocking(command_t(rank, command_t::INIT, 0, "")) > 0;
    DBG("VELOC initialized");
}

client_impl_t::client_impl_t(MPI_Comm c, const std::string &cfg_file) :
    cfg(cfg_file, false), comm(c), collective(true), gpu_cache(GPU_MEM), host_cache(HOST_MEM) {
    int provided;
    bool threaded = cfg.get_optional("threaded", false);
    MPI_Comm_rank(comm, &rank);
    if (threaded) {
        MPI_Query_thread(&provided);
        if (provided != MPI_THREAD_MULTIPLE)
            FATAL("MPI threaded mode requested but not available, please use MPI_Init_thread with the MPI_THREAD_MULTIPLE flag");
    }
    if (cfg.is_sync() || threaded) {
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
        MPI_Comm_rank(local, &provided);
        MPI_Comm_split(comm, provided == 0 ? 0 : MPI_UNDEFINED, rank, &backends);
        if (provided == 0)
            start_main_loop(cfg, backends, false);
        MPI_Barrier(local);
    } else
        launch_backend(cfg_file);
    queue = new comm_client_t<command_t>(rank);
    checkCuda(cudaGetDevice(&gpu_id));
    float host_cache_sz = std::stof(cfg.get("host_cache_size"));
    float gpu_cache_sz = std::stof(cfg.get("gpu_cache_size"));
    bool score_eviction = cfg.get_optional("score_eviction", false);
    size_t free_gpu_memory, total_gpu_memory;    
    checkCuda(cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory));
    total_gpu_cache = (size_t)((1<<30)*gpu_cache_sz);
    total_host_cache = (size_t)((1<<30)*host_cache_sz);
    DBG("Rank " << rank << "] Host cache: (MB) " << (total_host_cache >> 20) << " GPU cache: " << (total_gpu_cache >> 20) << " Total GPU memory: " << (total_gpu_memory >> 20));
    total_gpu_cache = free_gpu_memory >= total_gpu_cache ? total_gpu_cache : free_gpu_memory; 
    bool premature_eviction = cfg.get_optional("premature_eviction", false);
    host_cache.init(total_host_cache, -1, score_eviction, premature_eviction);
    gpu_cache.init(total_gpu_cache, gpu_id, score_eviction, premature_eviction);

    ec_active = run_blocking(command_t(rank, command_t::INIT, 0, "")) > 0;
    checkCuda(cudaStreamCreateWithFlags(&veloc_stream, cudaStreamNonBlocking));
    checkCuda(cudaStreamCreateWithFlags(&veloc_gpu_to_host_stream, cudaStreamNonBlocking));
    checkCuda(cudaStreamCreateWithFlags(&veloc_rec_stream, cudaStreamNonBlocking));

    write_to_file_thread = std::thread([&] { mem_to_file_trf(); });
    gpu_memcpy_thread = std::thread([&] { gpu_to_host_trf(); });
    prefetch_thread = std::thread([&] { prefetch_restart(); });
    
    write_to_file_thread.detach();
    gpu_memcpy_thread.detach();
    prefetch_thread.detach();

    trf_metrics["AtoD"] = 0;
    trf_metrics["DtoH"] = 0;
    trf_metrics["HtoF"] = 0;
    trf_metrics["FtoP"] = 0;

    trf_metrics["PtoF"] = 0;
    trf_metrics["FtoH"] = 0;
    trf_metrics["HtoD"] = 0;
    trf_metrics["DtoA"] = 0;
    trf_metrics["HtoA"] = 0;

    DBG("VELOC initialized (VELOC v1.5)");
}

client_impl_t::~client_impl_t() {
    if (collective && cfg.get_optional("threaded", false))
        MPI_Barrier(local);
    delete queue;
    delete modules;

    checkCuda(cudaStreamSynchronize(veloc_gpu_to_host_stream));
    checkCuda(cudaStreamSynchronize(veloc_rec_stream));
    checkCuda(cudaStreamSynchronize(veloc_stream));
    veloc_client_active = false;
    gpu_write_cv.notify_all();
    file_write_cv.notify_all();
    prefetch_cv.notify_all(); 
    checkCuda(cudaStreamDestroy(veloc_stream));
    checkCuda(cudaStreamDestroy(veloc_gpu_to_host_stream));
    checkCuda(cudaStreamDestroy(veloc_rec_stream));
    if (collective && cfg.get_optional("threaded", false)) {
        MPI_Barrier(local);
        MPI_Comm_free(&local);
    }
    DBG("[Rank: " << rank << "] " << "VELOC finalized");
}

bool client_impl_t::checkpoint_wait() {
    if (cfg.is_sync())
	    return true;
    // Wait for the checkpoint command queue to become empty.
    std::unique_lock<std::mutex> lock(ckpt_mutex);
    while(gpu_write_q.size() > 0 || file_write_q.size() > 0) 
        ckpt_cv.wait(lock);
    lock.unlock();

    if (checkpoint_in_progress) {
        ERROR("need to finalize local checkpoint first by calling checkpoint_end()");
        return false;
    }
    return queue->wait_completion() == VELOC_SUCCESS;
}

bool client_impl_t::mem_protect(int id, void *ptr, size_t count, size_t base_size, unsigned int flags=0, release_routine r_routine=NULL ) {
    if (id < 0 || id > MAX_REGIONS_PER_CKPT) {
        ERROR("ID of a region must be between 0 and " << MAX_REGIONS_PER_CKPT);
        return false;
    }
    mem_region_t *r = new mem_region_t(id, ptr, count*base_size, flags, r_routine, rank);
    mem_regions[id] = *r;
    return true;
}

bool client_impl_t::checkpoint(const std::string &name, int version) {
    if (version < 0 || version > MAX_VERSIONS_PER_SHOT) {
        ERROR("Version number per unique checkpoint name between 0 and " << MAX_VERSIONS_PER_SHOT);
        return false;
    }
    return checkpoint_begin(name, version) 
            && checkpoint_mem(VELOC_CKPT_ALL, {});
}

bool client_impl_t::checkpoint_begin(const std::string &name, int version) {
    if (checkpoint_in_progress) {
        ERROR("nested checkpoints not yet supported");
        return false;
    }
    if (!validate_name(name) || version < 0) {
        ERROR("checkpoint name and/or version incorrect: name can only include [a-zA-Z0-9_] characters, version needs to be non-negative integer");
        return false;
    }

    DBG("called checkpoint_begin for version " << version);
    current_ckpt = command_t(rank, command_t::CHECKPOINT, version, name.c_str());
    checkpoint_in_progress = true;
    if(ckpt_meta.find(version) != ckpt_meta.end()) {
        if(num_regions.find(version) != num_regions.end() && num_regions[version] > 0) {
            ERROR("Checkpointing of version " << version << " is still in progress");
            return false;
        }
        ckpt_meta.clear();
        num_regions.clear();
        trf_metrics.clear();
        trf_metrics["AtoD"] = 0;
        trf_metrics["DtoH"] = 0;
        trf_metrics["HtoF"] = 0;
        trf_metrics["FtoP"] = 0;

        trf_metrics["PtoF"] = 0;
        trf_metrics["FtoH"] = 0;
        trf_metrics["HtoD"] = 0;
        trf_metrics["DtoA"] = 0;
        trf_metrics["HtoA"] = 0;

    }
    return true;
}

bool client_impl_t::checkpoint_mem(int mode, const std::set<int> &ids) {
    cudaPointerAttributes attributes;
    void *ptr; size_t sz; int region_id; unsigned int flags=0;
    int ckpt_version = current_ckpt.version;
    
    if (!checkpoint_in_progress) {
        ERROR("must call checkpoint_begin() first");
        return false;
    }
    ckpt_regions.clear();
    ckpt_regions = mem_regions;
    size_t file_offset=0;
    std::unique_lock<std::mutex> ckpt_lock(ckpt_mutex);
    num_regions[ckpt_version] = ckpt_regions.size();
    ckpt_lock.unlock();
    for (auto &e : ckpt_regions) {     
        sz = e.second.sz;
        ptr = e.second.ptr;
        flags = e.second.flags;
        region_id = e.first;
        if (sz > total_host_cache || sz > total_gpu_cache) {
            ERROR("[Rank: " << rank << "] " << " Please increase cache size! Region: " << e.first << " needs " 
                << sz << " but available host " << total_host_cache << " available GPU: " << total_gpu_cache);
            return false;
        }
        mem_region_t *m = new mem_region_t(current_ckpt.name, ckpt_version, region_id, ptr, sz, file_offset);
        ckpt_meta[ckpt_version].insert(std::make_pair(e.first, sz));
        checkCuda(cudaPointerGetAttributes (&attributes, e.second.ptr));

        if (attributes.type==cudaMemoryTypeDevice) {
            if(flags == READ_ONLY) { 
                std::unique_lock<std::mutex> gpu_lock(gpu_write_mutex);
                gpu_write_q.push(m);
                gpu_lock.unlock();
                gpu_write_cv.notify_one();
            }
            else {                                      // Default region, create device copy.
                m->ptr = gpu_cache.get_mem_offset(m);
                checkCuda(cudaMemcpyAsync(m->ptr, ptr, sz, cudaMemcpyDeviceToDevice, veloc_stream));
                checkCuda(cudaStreamSynchronize(veloc_stream));
                trf_metrics["AtoD"] += sz;
                std::unique_lock<std::mutex> gpu_lock(gpu_write_mutex);
                gpu_write_q.push(m);
                gpu_lock.unlock();
                gpu_write_cv.notify_one();
            }
        } else { 
            std::unique_lock<std::mutex> file_lock(file_write_mutex);
            file_write_q.push(m);
            file_lock.unlock();
            file_write_cv.notify_one();
        }
        file_offset += sz;
    }
    checkpoint_in_progress = false;
    return true;
}

void client_impl_t::gpu_to_host_trf() {
    checkCuda(cudaSetDevice(gpu_id));
    mem_region_t *g, *h;
    while(true) {
        std::unique_lock<std::mutex> gpu_lock(gpu_write_mutex);
        while(gpu_write_q.size() == 0 && veloc_client_active)
            gpu_write_cv.wait(gpu_lock);
        if(!veloc_client_active)
            return;
        g = gpu_write_q.front();
        gpu_write_q.pop();
        gpu_lock.unlock();
        // WARN: This will not work when a single item is prefetched more than once
        if (cfg.get_optional("premature_eviction", false) && (g->pf_status == PREFETCH_CONSUMED || g->pf_status == PREFETCH_COMPLETED)) {
            gpu_cache.set_trf(g->uid, TRF_COMPLETED);
            continue;
        }
        h = new mem_region_t(g);
        host_cache.get_mem_offset(h);
        checkCuda(cudaMemcpyAsync(h->ptr, g->ptr, g->sz, cudaMemcpyDeviceToHost, veloc_gpu_to_host_stream));
        checkCuda(cudaStreamSynchronize(veloc_gpu_to_host_stream));
        trf_metrics["DtoH"] += g->sz;
        gpu_cache.set_trf(g->uid, TRF_COMPLETED);
        std::unique_lock<std::mutex> file_lock(file_write_mutex);
        file_write_q.push(h);
        file_lock.unlock();
        file_write_cv.notify_one();
    }    
    return;
}

void client_impl_t::mem_to_file_trf() {
    checkCuda(cudaSetDevice(gpu_id));
    size_t offset;

    do {
        std::ofstream file_stream; 
        try { 
            std::unique_lock<std::mutex> file_lock(file_write_mutex);
            while(file_write_q.size() == 0 && veloc_client_active) {
                file_write_cv.wait(file_lock);
            }
            file_lock.unlock();
            if(!veloc_client_active)
                return;
            mem_region_t *m = file_write_q.front();
            file_write_q.pop();
            // // WARN: This will not work when a single item is prefetched more than once
            mem_region_t* found = host_cache.search(m->uid);
            if (cfg.get_optional("premature_eviction", false) && (found == nullptr || found->pf_status == PREFETCH_CONSUMED)) {
                host_cache.set_trf(m->uid, TRF_COMPLETED);
                continue;
            }
            file_stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            command_t ckpt_command = command_t(rank, command_t::CHECKPOINT, m->ckpt_version, m->ckpt_name);
            file_stream.open(ckpt_command.filename(cfg.get("scratch")), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

            offset = sizeof(size_t) + ckpt_meta[m->ckpt_version].size()*(sizeof(size_t)+sizeof(int)) + m->f_offset;
            file_stream.seekp(offset);
            file_stream.write((char *)m->ptr, m->sz);
            DBG("Checkpoint version " << m->ckpt_version << " written to Scratch, setting trf_completed");
            host_cache.set_trf(m->uid, TRF_COMPLETED);
            trf_metrics["HtoF"] += m->sz;
            std::unique_lock<std::mutex> ckpt_lock(ckpt_mutex);  
            int num_reg = --num_regions[m->ckpt_version];
            ckpt_lock.unlock();
            if(num_reg == 0) {
                // Write headers
                file_stream.seekp(0);
                size_t regions_size = ckpt_meta[ckpt_command.version].size();
                file_stream.write((char *)&regions_size, sizeof(size_t));
                for (auto &e : ckpt_meta[ckpt_command.version]) {
                    file_stream.write((char *)&(e.first), sizeof(int));
                    file_stream.write((char *)&(e.second), sizeof(size_t));
                }
                if(!cfg.get_optional("disable_persistence", false)) {
                    checkpoint_end(ckpt_command);
                    trf_metrics["FtoP"] += m->sz;
                }
                ckpt_lock.lock();
                num_regions.erase(ckpt_command.version);
                ckpt_lock.unlock();
                ckpt_cv.notify_all();
            } 
            file_stream.close();
        } catch (std::ofstream::failure &f) {
            ERROR("cannot write to checkpoint file: reason: " << f.what());
            file_stream.close();
            return;
        }
    } while(true); 
}

bool client_impl_t::checkpoint_end(command_t current_ckpt) {
    if (cfg.is_sync()) {
        return modules->notify_command(current_ckpt) == VELOC_SUCCESS;
    }
    else if(veloc_client_active) {
        queue->enqueue(current_ckpt);
        return true;
    }
    return true;
}

bool client_impl_t::checkpoint_end(bool /*success*/) {
    checkpoint_in_progress = false;
    if (cfg.is_sync())
	    return modules->notify_command(current_ckpt) == VELOC_SUCCESS;
    else {
	    queue->enqueue(current_ckpt);
	    return true;
    }
}

int client_impl_t::run_blocking(const command_t &cmd) {
    if (cfg.is_sync())
	    return modules->notify_command(cmd);
    else {
	    queue->enqueue(cmd);
	    return queue->wait_completion();
    }
}

int client_impl_t::restart_test(const std::string &name, int needed_version) {
    if (!validate_name(name) || needed_version < 0) {
	ERROR("checkpoint name and/or version incorrect: name can only include [a-zA-Z0-9_] characters, version needs to be non-negative integer");
	return VELOC_FAILURE;
    }
    int version = run_blocking(command_t(rank, command_t::TEST, needed_version, name.c_str()));
    DBG(name << ": latest version = " << version);
    // if (collective) {
	//     int max_version;
    //     DBG("[Rank: " << rank << "] " << "Waiting at allreduce");
    //     MPI_Allreduce(&version, &max_version, 1, MPI_INT, MPI_MAX, comm);
	//     return max_version;
    // } else
	    return version;
}

std::string client_impl_t::route_file(const std::string &original) {
    char abs_path[PATH_MAX + 1];
    if (original[0] != '/' && getcwd(abs_path, PATH_MAX) != NULL)
	current_ckpt.assign_path(current_ckpt.original, std::string(abs_path) + "/" + original);
    else
	current_ckpt.assign_path(current_ckpt.original, original);
    return current_ckpt.filename(cfg.get("scratch"));
}

bool client_impl_t::next_prefetched(const char *name, int version, int region_id) {
    unsigned long long int uid = get_region_uid(std::string(name), version, region_id);
    auto it = gpu_cache.search(uid);
    if (it != nullptr && it->pf_status == PREFETCH_COMPLETED)
        return true;
    return false;
}

int client_impl_t::prefetch_enqueue(const char *name, int needed_version, int region_id) {
    // DBG("Enqueuing prefetch " << needed_version);
    mem_region_t *m = new mem_region_t((char *)name, needed_version, region_id);
    std::unique_lock<std::mutex> prefetch_lock(prefetch_mutex);
    if(prefetch_map.find(m->uid) != prefetch_map.end()) {
        std::queue<unsigned long long int> empty;
        std::swap(prefetch_order, empty);
        std::swap(prefetch_order_restart, empty);
        prefetch_map.clear();
    }
    prefetch_order.push(m->uid);
    prefetch_order_restart.push(m->uid);
    prefetch_map.insert(std::pair<unsigned long long int, mem_region_t *>(m->uid, m)); 
    prefetch_lock.unlock();
    gpu_cache.add_prefetch(m->uid);
    host_cache.add_prefetch(m->uid);
    start_prefetching = false;
    return VELOC_SUCCESS;
}

int client_impl_t::prefetch_start() {
    if(start_prefetching) {
        ERROR("Prefetching already in progress");
        return VELOC_FAILURE;
    }
    start_prefetching = !!prefetch_order.size();
    prefetch_cv.notify_all();
    return VELOC_SUCCESS;
}

void client_impl_t::prefetch_restart() {
    checkCuda(cudaSetDevice(gpu_id));
    while(veloc_client_active) {
        std::unique_lock<std::mutex> prefetch_lock(prefetch_mutex);
        while(veloc_client_active && !start_prefetching)
            prefetch_cv.wait(prefetch_lock);
        if(!veloc_client_active)
            return;
        unsigned long long int uid = prefetch_order.front();
        prefetch_order.pop();
        prefetch_lock.unlock();
        start_prefetching = !!prefetch_order.size();
        prefetch_lock.lock();
        prefetch_map[uid]->pf_status = PREFETCH_STARTED;
        prefetch_lock.unlock();
        prefetch_cv.notify_one();

        mem_region_t *f= gpu_cache.search(uid, PREFETCH_COMPLETED);
        if(f != nullptr) {
            DBG("Found checkpoint version " << f->ckpt_version << " on GPU cache");
            prefetch_lock.lock();
            prefetch_map[uid]->pf_status = PREFETCH_COMPLETED;
            prefetch_lock.unlock();
            prefetch_cv.notify_one();
            host_cache.del_prefetch(uid);
            continue;
        } 
        f = host_cache.search(uid, PREFETCH_COMPLETED);
        if(f != nullptr) {
            DBG("Found checkpoint version " << f->ckpt_version << " on Host cache");
            mem_region_t *r = new mem_region_t(f->ckpt_name, f->ckpt_version, f->region_id, f->ptr, f->sz, f->f_offset);
            gpu_cache.get_mem_offset(r);
            if (!veloc_client_active)
                return;
            checkCuda(cudaMemcpyAsync(r->ptr, f->ptr, f->sz, cudaMemcpyHostToDevice, veloc_rec_stream));
            checkCuda(cudaStreamSynchronize(veloc_rec_stream));
            trf_metrics["HtoD"] += f->sz;
            gpu_cache.set_trf(uid, TRF_COMPLETED);
            gpu_cache.set_pf(uid, PREFETCH_COMPLETED);
            host_cache.del_prefetch(uid);

            prefetch_lock.lock();
            prefetch_map[uid]->pf_status = PREFETCH_COMPLETED;
            prefetch_lock.unlock();
            prefetch_cv.notify_one();
            continue;
        }
        DBG("Found checkpoint name " << prefetch_map[uid]->ckpt_name << " version " << prefetch_map[uid]->ckpt_version << " on File");
        int result, end_result;
        pf_cmd = command_t(rank, command_t::RESTART, prefetch_map[uid]->ckpt_version, prefetch_map[uid]->ckpt_name);
        result = run_blocking(pf_cmd);
        end_result = result;
        header_size_pf = 0;
        if (end_result == VELOC_FAILURE) {
            ERROR("In prefetch_restart, end_result is not VELOC_SUCCESS for " << prefetch_map[uid]->ckpt_version << " got end result as: " << end_result);
            continue;
        } 
        recover_mem_prefetch(uid, end_result);
    }
}


bool client_impl_t::recover_mem_prefetch(unsigned long long int uid, int res) {
    if (prefetch_map.find(uid) == prefetch_map.end()) {
        ERROR("Cannot find region " << uid << " in prefetch map");
        return false;
    }
    if (header_size_pf == 0 && !read_header(true)) {
        ERROR("cannot recover in memory mode if header unavailable or corrupted");
        return false;
    }
    try {
        mem_region_t *m = prefetch_map[uid];
        std::ifstream f;
        f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        f.open(pf_cmd.filename(cfg.get("scratch")), std::ifstream::in | std::ifstream::binary);
        size_t rec_size = 0, file_offset = header_size_pf; 
        auto it = region_info_pf.find(m->region_id);
        if(it == region_info_pf.end()) {
            ERROR("Region " << m->region_id << " not found in file during recovery");
            return false;
        }
        rec_size = region_info_pf[m->region_id];
        file_offset += region_f_offset[m->region_id];
        if(rec_size > total_host_cache || rec_size > total_gpu_cache) {
            ERROR("Prefetch cannot be completed for checkpoint " << m->ckpt_name << " version " << m->ckpt_version
                << " required cache: " << rec_size << " avail. host cache: " << total_host_cache << " avail. gpu cache" 
                << total_gpu_cache << ", please increase host and/or GPU cache!");
            return VELOC_FAILURE;
        }
        m->sz = rec_size;
        f.seekg(file_offset);
        mem_region_t *h = new mem_region_t(m->ckpt_name, m->ckpt_version, m->region_id, m->sz);
        host_cache.get_mem_offset(h);
        host_cache.set_pf(h->uid, PREFETCH_STARTED);
        f.read((char*)h->ptr, rec_size);      
        if (res == 100)
            trf_metrics["PtoF"] += h->sz;
        trf_metrics["FtoH"] += h->sz;
        host_cache.set_trf(h->uid, TRF_COMPLETED);
        host_cache.set_pf(h->uid, PREFETCH_COMPLETED);
        DBG("Read from file to host cache for " << m->ckpt_name << " version " << m->ckpt_version);
        
        mem_region_t *g = new mem_region_t(m->ckpt_name, m->ckpt_version, m->region_id, m->sz);
        gpu_cache.get_mem_offset(g);
        gpu_cache.set_pf(g->uid, PREFETCH_STARTED);
        checkCuda(cudaMemcpyAsync(g->ptr, h->ptr, rec_size, cudaMemcpyHostToDevice, veloc_rec_stream));
        checkCuda(cudaStreamSynchronize(veloc_rec_stream));
        trf_metrics["HtoD"] += h->sz;
        gpu_cache.set_trf(g->uid, TRF_COMPLETED);
        gpu_cache.set_pf(g->uid, PREFETCH_COMPLETED);
        host_cache.del_prefetch(h->uid);
        DBG("Read from host cache to gpu cache for " << m->ckpt_name << " version " << m->ckpt_version);

        std::unique_lock<std::mutex> prefetch_lock(prefetch_mutex);
        prefetch_map[m->uid]->pf_status = PREFETCH_COMPLETED;
        prefetch_lock.unlock();
        prefetch_cv.notify_all(); 
    } catch (std::ifstream::failure &e) {
        ERROR("cannot read checkpoint file " << pf_cmd << ", reason: " << e.what());
        return false;
    }
    return true;
}


bool client_impl_t::restart(const std::string &name, int version) {
    return restart_begin(name, version)
        && recover_mem(VELOC_CKPT_ALL, {})
        && restart_end(true);
}

bool client_impl_t::restart_begin(const std::string &name, int version) {
    if (checkpoint_in_progress) {
	    INFO("cannot restart while checkpoint in progress");
	    return false;
    }
    if (!validate_name(name) || version < 0) {
	    ERROR("checkpoint name and/or version incorrect: name can only include [a-zA-Z0-9_] characters, version needs to be non-negative integer");
	    return VELOC_FAILURE;
    }
    current_ckpt = command_t(rank, command_t::RESTART, version, name.c_str());
    // HACK: FIXME: Prefetch maps are addressed by uid, but we do not have that here.
    // So we consider every checkpoint version has region 0 enqueued to be prefetched.
    unsigned long long int uid = get_region_uid(name, version, 0);
    if(prefetch_map.find(uid) != prefetch_map.end())
        return true;
    
    int result, end_result;
    result = run_blocking(current_ckpt);
    end_result = result;
    if (end_result != VELOC_FAILURE) {
        header_size = 0;
	    return true;
    } else
	    return false;
}

bool client_impl_t::read_header(bool is_pf) {
    if (is_pf) {
        region_info_pf.clear();
        region_f_offset.clear();
    } else
        region_info.clear();
    try {
        std::ifstream f;
        size_t expected_size = 0;
        f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        f.open((is_pf ? pf_cmd : current_ckpt).filename(cfg.get("scratch")), std::ifstream::in | std::ifstream::binary);
        size_t no_regions, region_size;
        int id;
        f.read((char *)&no_regions, sizeof(size_t));
        for (unsigned int i = 0; i < no_regions; i++) {
            f.read((char *)&id, sizeof(int));
            f.read((char *)&region_size, sizeof(size_t));
            if (is_pf) {
                region_f_offset.insert(std::make_pair(id, expected_size));
                region_info_pf.insert(std::make_pair(id, region_size));
            } else 
                region_info.insert(std::make_pair(id, region_size));
            expected_size += region_size;
        }
        if (is_pf)
            header_size_pf = f.tellg();
        else
            header_size = f.tellg();
        f.seekg(0, f.end);
        size_t file_size = (size_t)f.tellg() - header_size;
        if (is_pf)
            file_size = (size_t)f.tellg() - header_size_pf;
        if (file_size != expected_size)
            throw std::ifstream::failure("file size " + std::to_string(file_size) + " does not match expected size " + std::to_string(expected_size));
    } catch (std::ifstream::failure &e) {
        ERROR("cannot validate header for checkpoint " << (is_pf ? pf_cmd : current_ckpt) << ", reason: " << e.what());
        header_size = 0;
        header_size_pf = 0;
        return false;
    }
    return true;
}

size_t client_impl_t::recover_size(char *ckpt_name, int ckpt_version, int region_id) {
    if(ckpt_meta.find(ckpt_version) != ckpt_meta.end()) {
        auto it = ckpt_meta[ckpt_version].find(region_id);
        if (it != ckpt_meta[ckpt_version].end())
            return it->second;
    }
    unsigned long long int uid = get_region_uid(std::string(ckpt_name), ckpt_version, region_id);
    mem_region_t *f = gpu_cache.search(uid);
    if (f != nullptr)
        return f->sz;
    f = host_cache.search(uid);
    if (f != nullptr)
        return f->sz;
    // Read the size from the file.
    current_ckpt = command_t(rank, command_t::RESTART, ckpt_version, ckpt_name);
    return recover_size(region_id);
}

size_t client_impl_t::recover_size(int id) {
    if (header_size == 0)
        read_header();
    auto it = region_info.find(id);
    if (it == region_info.end())
	return 0;
    else
	return it->second;
}

bool client_impl_t::recover_mem(int mode, const std::set<int> &ids) {   
    // TODO: If ckpt_meta is empty, i.e. after crash, read file.
    std::set<int> pending = {};
    unsigned long long int uid;
    for(auto &e: ckpt_meta[current_ckpt.version]) {
        bool found = ids.find(e.first) != ids.end();
        if ((mode == VELOC_RECOVER_SOME && !found) || (mode == VELOC_RECOVER_REST && found))
            continue;
        if (mem_regions.find(e.first) == mem_regions.end()) {
            ERROR("no protected memory region defined for id " << e.first);
            return false;
        }
        if (mem_regions[e.first].sz < e.second) {
            ERROR("protected memory region " << e.first << " is too small ("
                << mem_regions[e.first].sz << ") to hold required size ("
                << e.second << ")" << " for checkpoint " << current_ckpt.version);
            return false;
        }
        uid = get_region_uid(std::string(current_ckpt.name), current_ckpt.version, e.first);
        found = prefetch_map.find(uid) != prefetch_map.end();
        if(found) {
            unsigned long int front_pf = prefetch_order_restart.front();
            prefetch_order_restart.pop();
            if (front_pf != uid) {
                // DBG("Penalizing in restart mem for " << current_ckpt.version);
                pending.insert(e.first);
                prefetch_map.erase(front_pf);
                continue;
            }
            std::unique_lock<std::mutex> prefetch_lock(prefetch_mutex);
            while (prefetch_map[uid]->pf_status != PREFETCH_COMPLETED) 
                prefetch_cv.wait(prefetch_lock);
            prefetch_lock.unlock();
            mem_region_t *g = gpu_cache.search(uid);
            checkCuda(cudaMemcpyAsync((void *)mem_regions[e.first].ptr, g->ptr, e.second, cudaMemcpyDeviceToDevice, veloc_stream));
            checkCuda(cudaStreamSynchronize(veloc_stream));
            trf_metrics["DtoA"] += g->sz;
            gpu_cache.del_prefetch(uid);
            host_cache.del_prefetch(uid);
            prefetch_lock.lock();
            prefetch_map.erase(uid);
            prefetch_lock.unlock();
        } else
            pending.insert(e.first);
    }
    if(pending.size()) {
        for (auto region_id: pending) {
            uid = get_region_uid(std::string(current_ckpt.name), current_ckpt.version, region_id);
            mem_region_t *f = gpu_cache.search(uid, PREFETCH_COMPLETED);
            if (f != nullptr) {
                DBG("Found " << current_ckpt.version << " on GPU");
                checkCuda(cudaMemcpy(mem_regions[region_id].ptr, f->ptr, f->sz, cudaMemcpyDeviceToDevice));
                trf_metrics["DtoA"] += f->sz;
                gpu_cache.del_prefetch(f->uid);
                continue;
            } 
            f = host_cache.search(uid, PREFETCH_COMPLETED);
            if (f != nullptr) {
                DBG("Found " << current_ckpt.version << " on Host");
                checkCuda(cudaMemcpy(mem_regions[region_id].ptr, f->ptr, f->sz, cudaMemcpyHostToDevice));
                trf_metrics["HtoA"] += f->sz;
                host_cache.del_prefetch(f->uid);
                continue;
            }
            DBG("Could not find " << current_ckpt.version << " region id: " << region_id << " on host or GPU cache");
            if (!read_header() && header_size == 0) {
                ERROR("cannot recover in memory mode if header unavailable or corrupted");
                return false;
            }
            try {
                std::ifstream f;
                f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                f.open(current_ckpt.filename(cfg.get("scratch")), std::ifstream::in | std::ifstream::binary);
                f.seekg(header_size);
                for (auto &e : region_info) {
                    if (pending.find(e.first) == pending.end())
                        continue;
                    cudaPointerAttributes attributes;
                    checkCuda(cudaPointerGetAttributes (&attributes, mem_regions[e.first].ptr));
                    if (attributes.type==cudaMemoryTypeDevice) {
                        mem_region_t *m = new mem_region_t(current_ckpt.name, current_ckpt.version, e.first, e.second);
                        void *h_ptr = host_cache.get_mem_offset(m);
                        f.read((char *)h_ptr, e.second);
                        trf_metrics["FtoH"] += e.second;
                        checkCuda(cudaMemcpy(mem_regions[e.first].ptr, h_ptr, e.second, cudaMemcpyHostToDevice));
                        trf_metrics["HtoA"] += e.second;
                        host_cache.set_trf(m->uid, TRF_COMPLETED);
                    }
                    else 
                        f.read((char *)mem_regions[e.first].ptr, e.second);
                }
            } catch (std::ifstream::failure &e) {
                ERROR("cannot read checkpoint file " << current_ckpt << ", reason: " << e.what());
                return false;
            }
        }
    }
    return true;
}

bool client_impl_t::restart_end(bool /*success*/) {
    return true;
}


void client_impl_t::get_trf_metrics(char *c) {
    std::string s = "{";
    for(const auto& e : trf_metrics) {
        // std::cout << e.first << ": " << e.second << std::endl;
        s += std::string(e.first+" : "+std::to_string(e.second)+", ");
    }
    s += "}";
    strcpy(c, s.c_str());
}