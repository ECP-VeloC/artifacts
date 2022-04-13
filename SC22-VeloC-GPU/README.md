# VeloC-GPU
Tests for evaluating VELOC-GPU prefetching techinques

## Installing VeloC-GPU:
1. git clone https://github.com/ECP-VeloC/artifacts.git
2. cd SC22-VeloC-GPU/veloc-gpu
3. ./bootstrap.sh
4. ./auto-install.py <veloc-build-dir>

Export the following variables
```
export LD_LIBRARY_PATH=<veloc-build-dir>/lib
export PATH=<veloc-build>/bin:$PATH
```

To run evaluation, follow [scripts/README.md](./scripts//README.md).

## Metrics of evaluation
We aim to study the influence of following factors on the waiting time of the application while restoring a given memory region (checkpoint):
1. Size of GPU cache (_from config file_)
2. Size of Host cache (_from config file_)
3. Eviction approaches  (_from config file_)
    - FIFO eviction (drop the oldest memory region on the cache)
    - Score-based sliding window (assigning scores to memory region based on factors such as time to flush and prefetch hints)
4. Premature eviction (_from config file_): Allow evicting a region from GPU or Host cache if it is consumed (restored) by the application before it gets transferred to lower memory tiers. 
This saves bandwidth of transferring the checkpoint from GPU to host and host to SSD, but produces garbage values of the restored memory regions in lower tiers.
5. Number of prefetches enqueued  (_application parameter_)
    - No prefetches
    - Single prefetch at a time
    - Multiple prefetches at beginning of a shot

### Configuration
An example configuration file: [sample-config.cfg](./scripts/config.cfg).
We assume that all the checkpoints of the application can be accomodated on local SSD and does not need to be transferred to remote (persistent) storage. 
Also, we assume that there exists a 1:1 mapping between GPUs and MPI processes.
1. `scratch`: Points to local SSD directory (or tmpfs if it is large enough for all checkpoints)
2. `persistent`: Points to a remote directory (mandatory field in configuration, but will not be used if `persistent_interval` is set to a high number)
3. `persistent_interval`: Time to wait before flushing from scratch to persistent storage (in seconds). We set this to a very high number to avoid read-write contention from scratch directory while restore operations.
4. `gpu_cache_size`: GPU memory to be allocated and used by VELOC only (in GB) per GPU
5. `host_cache_size`: Host memory to be allocated and used by VELOC only per process
6. `score_eviction`: (true) Set the eviction policy to score-based sliding window approach, (false) use FIFO eviction policy
7. `premature_eviction`: (false) Allows pre-mature eviction of a memory region after it has been consumed but not necessarily flushed to lower tiers.

### Input params
1. Configuration file: Required config file [check configuration section](#configuration)
2. Checkpoint/Restore interval (in ms): Will sleep for this interval instead of performing computations, _RTM should run kernels instead_.
3. Size: Integer buffer of size 2^(size), _RTM will have different sizes for each snapshot as opposed to a static size_
4. Number of checkpoints per shot
5. Number of shots
6. Prefetch/Restore pattern: 
    - (0) Ascending order
    - (1) Reverse order
    - (2) Random order
7. Number of prefetches enqueued

<!-- 
### Flow of API calls
1. Init MPI in multi-threaded mode.
2. Call `cudaSetDevice` before calling `VELOC_Init`
3. Call `VELOC_Mem_protect` to mark application buffers which need to be checkpointed.
4. Each shot should have a unique checkpoint name to avoid ambiguity while searching in the caches or scratch file.
5. If multiple prefetches are enabled, enqueue prefetch requests in required order using `VELOC_Prefetch_enqueue`.
6. Checkpoint using `VELOC_Checkpoint`
7. **Wait for checkpoints to be flushed** to SSD (to avoid dividing bandwidth between writing and reading during restore phase)
8. Start prefetching from lower tiers using `VELOC_Prefetch_start` call.
9. If single prefetching is enabled, enqueue and start prefetching the next memory region
10. Measure the time required for restore operation.
11. Check how many of the `next required memory regions` are already prefetched to the GPU cache using `VELOC_Next_prefetched` call, which returns `false` if a given version is still not completely prefetched to the GPU cache.
12. Display metrics (`next_prefetched` and `restore_time` 
-->

### Experiments
For different GPU and Host cache sizes, we run the following 6 experiments:
  | Input value       | Description                                           | `score_eviction` flag | 
  | :----------------:|:-----------------------------------------------------:|:---------------------:|
  | 0                 | No prefetches to be enqueued                          | true/false            |
  | 1                 | Single prefetch enqueued at a time                    | true/false            |
  | 2                 | Multiple prefetches enqueued in the beginning         | true/false            |
