start=`date +%s`

echo "In variable sizes run script"
dir=$PWD # Contains variable-sizes.cu
cd $dir
openmpi_build=/path/to/openmpi-4.0.5/
veloc_build=$HOME/veloc-build
traces=$dir/sample-trace.csv
input_file=$dir/variable-sizes.cu
exec_file=$dir/variable-sizes.out
# Compile the code
nvcc -std=c++17 -I$openmpi_build/include -L/$openmpi_build/lib -lmpi -I $veloc_build/include/ -I $veloc_build/bin -L$veloc_build/lib -lveloc-client $input_file -o $exec_file

ranks=(1 2 4 6 8)
num_shots=1
sleep_times=(10 15)
prefetch_orders=(0 1 2)
num_prefetches=(0 1 2)
score_eviction=(0 1)
veloc_wait=(1)
gpu_caches=(4)
host_caches=(32)

for host_cache in ${host_caches[@]}; do
    for gpu_cache in ${gpu_caches[@]}; do
        for num_ranks in ${ranks[@]}; do
            outputs=$dir/results/$num_ranks/variable-res-${gpu_cache}G${host_cache}H
            mkdir -p $outputs
            echo "Output in $outputs"            
            for s in ${sleep_times[@]}; do
                sed -i "s/host\\_cache\\_size.*/host\\_cache\\_size=${host_cache}/g" $config_file
                sed -i "s/gpu\\_cache\\_size.*/gpu\\_cache\\_size=${gpu_cache}/g" $config_file
                for po in ${prefetch_orders[@]}; do
                    for npf in ${num_prefetches[@]}; do
                        for se in ${score_eviction[@]}; do
                            echo "Starting on num ranks : "${num_ranks};
                            echo "Starting num shots : "${num_shots};
                            echo "Starting for sleep time $s";
                            echo "Starting for prefetch order $po";
                            echo "Starting for number of prefetches $npf";
                            echo "Starting for score eviction $se";
                            sed -i "s/score\\_eviction.*/score\\_eviction=false/g" $config_file
                            sed -i "s/eager\\_eviction.*/eager\\_eviction=false/g" $config_file
                            if (( $se == 1 ));
                            then
                                sed -i "s/score\\_eviction.*/score\\_eviction=true/g" $config_file
                            fi                        
                            curr_output=$outputs/res-variable-$num_ranks-$s-$po-$npf-$se.log
                            cat $config_file >> $curr_output
                            (time mpirun -hostfile $COBALT_NODEFILE -n $num_ranks -npernode 8 $exec_file $config_file $traces ${num_shots} $s $po $npf) >> $curr_output 2>&1
                        done
                    done
                done
            done
        done
    done
done

end=`date +%s`
runtime=$((end-start))
echo "Total runtime is $runtime"