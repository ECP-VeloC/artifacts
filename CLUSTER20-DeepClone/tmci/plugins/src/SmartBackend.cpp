#include "SmartBackend.hpp"

#include <cassert>
#include <limits>

TMCI_REGISTER_BACKEND(smart, SmartBackend);

static size_t THRESHOLD = (1 << 20);

void __attribute__ ((constructor)) smart_constructor() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &standby);
    subrank = rank / 2;
}

int SmartBackend::Save(int id, const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) {
    assert(id >=0 && tensors.size() == 1); // one tensor at a time
    unsigned int i = tcount++;
    if (i == 0) // send start signal
        MPI_Send(&no_tensors, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);

    const tensorflow::Tensor &t = tensors[0];
    size_t tsize = t.tensor_data().size();
    assert(tsize <= std::numeric_limits<int>::max()); // MPI limitation
    int chunk_size = tsize / size * 2;
    if (2 * tsize % size == 0 && tsize > THRESHOLD) { // can split tensor, send chunk
	//std::cerr << "rank " << rank << " sent tensor chunk for '" << id << "' of size '" << chunk_size << "'" << std::endl;
	MPI_Isend((char *)t.tensor_data().data() + subrank * chunk_size, chunk_size, MPI_BYTE, rank + 1, id, MPI_COMM_WORLD, &reqs[i]);
    } else { // can't split tensor, send it whole
        //std::cerr << "rank " << rank << " sent full tensor for '" << id << "' of size '" << tsize << "'" << std::endl;
        MPI_Isend((char *)t.tensor_data().data(), tsize, MPI_BYTE, rank + 1, id, MPI_COMM_WORLD, &reqs[i]);
    }
    if (tcount == no_tensors) {
        MPI_Waitall(no_tensors, reqs, MPI_STATUSES_IGNORE);
        //std::cerr << "rank " << rank << " wait successful, all tensors sent" << std::endl;
        tcount = 0;
    }

    return 0;
}

int SmartBackend::Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) {
    int recv_tensors = 0;
    MPI_Recv(&recv_tensors, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    assert(no_tensors == recv_tensors); // check if signal is valid

    MPI_Status status;
    unsigned long total_size = 0;
    double ts = MPI_Wtime();
    for (int i = 0; i < no_tensors; i++) {
        int count;
        MPI_Probe(rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &count);
        assert(status.MPI_TAG < no_tensors);
        const tensorflow::Tensor &t = tensors[status.MPI_TAG];
        size_t tsize = t.tensor_data().size();
        total_size += tsize;
        int chunk_size = tsize / size * 2;
        if (2 * tsize % size == 0 && tsize > THRESHOLD) { // can split tensor, receive chunk
             assert(count == chunk_size); 
             //std::cerr << "rank " << rank << " receiving tensor chunk for '" << status.MPI_TAG << "' of size '" << count << "'" << std::endl;
             MPI_Recv((char *)t.tensor_data().data() + subrank * chunk_size, chunk_size, MPI_BYTE, rank - 1, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else { // can't split tensor, receive whole
             assert(count == tsize);
             //std::cerr << "rank " << rank << " receiving full tensor for '" << status.MPI_TAG << "' of size '" << count << "'" << std::endl;
             MPI_Recv((char *)t.tensor_data().data(), tsize, MPI_BYTE, rank - 1, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    int sharded = 0;
    if (size > 2) {
        for (const tensorflow::Tensor &t : tensors) {
            size_t tsize = t.tensor_data().size();
            int chunk_size = tsize / size * 2;
            if (2 * tsize % size == 0 && tsize > THRESHOLD) {
                //std::cerr << "rank " << rank << " entering all_gather for tensor " << tcount << std::endl;
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, (char *)t.tensor_data().data(), chunk_size, MPI_BYTE, standby);
                sharded++;
            }
        }
    }
    fprintf(stderr, "rank %d, received %d (%d sharded) tensors (size = %lu MB) in %.3lf\n", rank, no_tensors, sharded, total_size >> 20, MPI_Wtime() - ts);

    return 0;
}
