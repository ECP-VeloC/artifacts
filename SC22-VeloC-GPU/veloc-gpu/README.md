# VELOC: VEry-Low Overhead Checkpointing System

VeloC is a multi-level checkpoint/restart runtime that delivers 
high performance and scalability for complex heterogeneous storage 
hierarchies without sacrificing ease of use and flexibility.

It is primarily used as a fault-tolerance tool for tightly coupled
HPC applications running on supercomputing infrastructure but is
essential in many other use cases: suspend-resume, migration, 
debugging.

VeloC is a collaboration between Argonne National Laboratory and 
Lawrence Livermore National Laboratory as part of the Exascale 
Computing Project.

## VELOC-GPU
The VELOC-GPU project is an experimental prototype for enabling checkpoint/restart
support on NVidia GPUs. The parent VELOC source is available at: https://github.com/ECP-VeloC/VELOC
and the VELOC-GPU source is available at: https://github.com/ECP-VeloC/artifacts/SC22-VeloC-GPU.

To run VELOC with GPU support, perform the following steps:
1. cd veloc-gpu
2. ./bootstrap.sh
3. ./auto-install.py <veloc-build-dir>

For checkpointing data residing on the GPU memory, use the following `VELOC_Mem_protect` syntax
`VELOC_Mem_protect(id, &device_var, var_length, sizeof(data_type), flag)`. Here, `flag` can be either of
`DEFAULT` or `READ_ONLY`. In the case of `DEFAULT` flag, VELOC_Checkpoint would relinquish control back 
to the application only when a copy of the original device variable is created, either on the GPU or the host,
whereas the `READ_ONLY` flag specifies that the device variable will not change until the checkpoint is completed,
and hence the control to the application is relinquished almost immediately.

To run the [gpu-checkpoint example](./test/gpu-checkpoint.cu),

Compile using: 
```
export LD_LIBRARY_PATH=<veloc-build>/lib
export PATH=<veloc-build>/bin:$PATH
nvcc -I /path/to/mpi/ -L /mpi/lib -lmpi -I /<veloc-build>/include/ -L /<veloc-build>/lib -lveloc-client gpu-checkpoint.cu -o gpu-checkpoint
```

Run using `mpirun -np 1 ./gpu-checkpoint ./test/gpu-checkpoint.cfg`

## Documentation

The documentation of VeloC is available here: http://veloc.rtfd.io

It includes a quick start guide as well that covers the basics needed
to use VeloC on your system.

## Contacts

In case of questions and comments or help, please contact the VeloC team at 
veloc-users@lists.mcs.anl.gov


## Release

Copyright (c) 2018-2020, UChicago Argonne LLC, operator of Argonne National Laboratory <br>
Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

For release details and restrictions, please read the [LICENSE](https://github.com/ECP-VeloC/VELOC/blob/master/LICENSE) 
and [NOTICE](https://github.com/ECP-VeloC/VELOC/blob/master/NOTICE) files.

`LLNL-CODE-751725` `OCEC-18-060`
