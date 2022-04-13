# Test scripts for running uniform and variable sized checkpoint simulation using VELOC-GPU

## To run tests:

We run the entire suite of tests (as outlined in [README](./../README.md)) with all combinations of eviction policy, number of prefetch operations, and restore order for a given value of host/GPU cache. 
The size of each checkpoint and restore intervals can be specified in the script in variables `sleep_times` and `sizes`.

After setting `openmpi_build=/path/to/openmpi`, `veloc_build=/path/to/veloc-build` and `dir=/path/to/test-scripts` each in all of the following files:
`uniform-run.sh`, `variable-run.sh` and `variable-run-scalability.sh`, they can be run to compile and generate the output for uniform, and variable sized checkpoints, and scalability for variable sized checkpoint respectively. The outputs are placed in `results` folder.

For plotting the graphs, use:

`python3 plot-wait-graphs.py <results-folder> <is_rtm=0|1>`: This generates Figures 5-8 as shown in the experiment section of our paper. The `is_rtm` parameter (default 0) can be set to 1 to generate graphs for uniform checkpoint sizes.
`python3 plot-scalability-graphs.py`: This generates Figure 9 as shown in the experiment section of our paper.