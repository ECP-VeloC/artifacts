#ifndef __MEM_REGION_HPP
#define __MEM_REGION_HPP

#include "include/veloc.h"
#include <deque>
#include <limits>
#include <climits>
#include <map>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExtCudaRt.h>

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

unsigned long long int get_region_uid(std::string ckpt_name, int version, int region_id);

struct mem_region_t
{
    char ckpt_name[PATH_MAX];
    int ckpt_version;
    int region_id;
    void *ptr = nullptr;
    size_t sz;
    int flags = -1;
    release_routine r_routine;
    int rank;
    size_t f_offset = std::numeric_limits<size_t>::max(); // File offset
    size_t start_offset = std::numeric_limits<size_t>::max();
    size_t end_offset = std::numeric_limits<size_t>::max();
    int trf_status = INIT_REGION;
    unsigned int release_at = 0;
    int pf_status = NO_PREFETCH; // Prefetch flag
    unsigned long long int uid;
    mem_region_t() : 
        ckpt_name("-1"), ckpt_version(-1), region_id(-1), ptr(nullptr), sz(0), flags(-1), rank(-1), f_offset(0), start_offset(0), end_offset(0), trf_status(-1), pf_status(-1){};
    mem_region_t(mem_region_t *m) {
        memcpy(this, m, sizeof(mem_region_t));
        this->pf_status = NO_PREFETCH;
        this->trf_status = INIT_REGION;
    }
    // Used in insert for memory cache function
    mem_region_t(char *n, int v, int r, size_t s, size_t e, int m) : 
        ckpt_version(v), region_id(r), start_offset(s), end_offset(e), trf_status(m) {
        strcpy(ckpt_name, n);
        sz = e-s;
        uid = get_region_uid(std::string(n), ckpt_version, region_id);
    };
    // Used in recover_mem function
    mem_region_t(char *n, int v, int r, size_t s) : 
        ckpt_version(v), region_id(r), sz(s) {
        strcpy(ckpt_name, n);
        uid = get_region_uid(std::string(n), ckpt_version, region_id);
    };
    // Used in the main ckpt function
    mem_region_t(char *n, int v, int r, void *p, size_t sz, size_t f) : 
        ckpt_version(v), region_id(r), ptr(p), sz(sz), f_offset(f) {
        strcpy(ckpt_name, n);
        uid = get_region_uid(std::string(n), ckpt_version, region_id);
    };
    // Used in prefetch
    mem_region_t(char *n, int v, int r): ckpt_version(v), region_id(r) {
        strcpy(ckpt_name, n);
        uid = get_region_uid(std::string(n), ckpt_version, region_id);
    };
    // Used in mem_protect
    mem_region_t(int id, void * p, size_t sz, int flag, release_routine r_routine, int r):
        region_id(id), ptr(p), sz(sz), flags(flag), r_routine(r_routine), rank(r) {}
};

typedef std::unordered_map<unsigned long long int, mem_region_t*> mem_map_t;
typedef std::deque<mem_region_t* > mem_records_t;
typedef std::tuple<size_t, double, mem_records_t> mem_set_t;

#endif // End of __MEM_REGION_HPP