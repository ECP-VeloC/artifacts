#ifndef __SMART_BACKEND_HPP
#define __SMART_BACKEND_HPP

#include <tmci/backend.hpp>
#include <mpi.h>
#include <atomic>

static int rank, size, subrank;
static MPI_Comm standby;

class SmartBackend : public tmci::Backend {
    unsigned int no_tensors;
    MPI_Request *reqs;
    std::atomic_uint32_t tcount{0};

public:
    SmartBackend(const char* config) {
        if (sscanf(config, "%u", &no_tensors) != 1)
            throw std::runtime_error(std::string("config passed to tensor op '") + config + "' is invalid");
        reqs = new MPI_Request[no_tensors];
        std::cerr << "rank " << rank << " initialized with config: '" << no_tensors << "'" << std::endl; 
    }
    ~SmartBackend() {
        delete []reqs;
    }
    virtual int Save(int id, const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors);
    virtual int Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors);
};

#endif
