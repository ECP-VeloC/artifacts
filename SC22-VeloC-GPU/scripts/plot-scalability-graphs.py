#!/usr/bin/python3
from xmlrpc.client import Boolean
import matplotlib.pyplot as plt
import sys
import numpy as np
from pathlib import Path
import re
import copy
plt.rcParams.update({'font.size': 15})

ranks = [8]
sizes = [27] #1<<x MB
num_shots = [1]
sleep_times = {}
sleep_times[27]=[10, 15]
sleep_times["variable"]=[10, 15]
prefetch_orders = [0, 1, 2]
num_prefetches = [0, 1, 2]
score_eviction=[0, 1]
pname = ["Same", "Reverse", "Random"]
gpu_cache_size = 4
host_cache_size = 32

def plot_scalability(meta, data):
    fifovariable = []
    scvariable = []
    labels = []
    max_v = 0
    for num_ranks in meta["ranks"]:
        labels.append(num_ranks)
        for s in meta["sleep_times"]:        
            fifovariable.append(data[num_ranks][s][0]["time"])
            scvariable.append(data[num_ranks][s][1]["time"])
            max_v = max(data[num_ranks][s][0]["time"], max_v)
            
    x = np.arange(len(labels)) 
    width = 0.20  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, fifovariable, width, label='0 PF, FIFO', edgecolor='black', hatch="-")
    ax.bar(x + width/2, scvariable, width, label='* PF, Score', edgecolor='black', hatch="*")
    ax.set_ylabel('Cumulative wait time per process (ms)', fontsize=15)
    ax.set_xlabel('Number of concurrent processes (GPUs)', fontsize=15)
    ax.set_yscale('log')
    ax.set_xticks(x, labels)
    ax.set_ylim(top=1.5*max_v)
    ax.legend(frameon= 0, ncol=2, loc='upper right', prop={'size': 14, 'weight': 'bold'})
    fig.tight_layout() 
    fig.savefig(f"uniform-scalability.pdf", format="pdf", bbox_inches='tight')
    fig.tight_layout()
    # plt.show()  
                

def get_host_gpu_cache(results_folder):
    n = results_folder.split("-")[-1]
    if(re.search("^\d+G\d+H", n)):
        global host_cache_size, gpu_cache_size
        host_cache_size = int(n.split("G")[-1].split("H")[0])
        gpu_cache_size = int(n.split("G")[0])


def process_time_agg(a, r):
    num_ckpts = len(a)//r
    t = [0]*num_ckpts
    for c in range(num_ckpts):
        for i in range(r):
            t[c] +=a[i*num_ckpts+c]
        t[c] = t[c]/r
    return t

def process_trf_agg(d, r):
    k = d[0].keys()
    res = {x: 0 for x in k}
    for x in k:
        for i in range(r):
            res[x] += d[i][x]
    return res

def convert_to_dict(s):
    s = s[1:-1]
    d = dict(map(str.strip, sub.split(':', 1)) for sub in s.split(', ') if ':' in sub)
    for k, v in d.items():
        d[k] = int(v)
    return d

def read_recs(num_ranks, lines, sz, po):
    pf_time = [[] for _ in range(num_ranks)]
    npf = [[] for _ in range(num_ranks)]
    all_pf_time = []
    all_npf = []
    trf_data = []
    start_index = -5
    for i in range(num_ranks):
        temp = [int(x)*(10**-6) for x in lines[start_index-num_ranks-i].split(", ")[:-1]]
        if (po == 1):
            temp.reverse()
        elif (po == 2):
            p_order = temp = [int(x) for x in lines[-3*num_ranks+start_index-num_ranks-i].split(", ")[:-1]]
            x = copy.deepcopy(temp)
            for j in range(len(x)):
                temp[j] = x[p_order[j]]            
        pf_time[i] = temp
        npf[i] = [int(x) for x in lines[-3*num_ranks+start_index-i].split(", ")[:-1]]
        trf_data.append(convert_to_dict(str(lines[start_index-i])))
        all_pf_time.extend(temp)
        all_npf.extend(npf[i])
    return process_time_agg(all_pf_time, num_ranks), process_time_agg(all_npf, num_ranks), process_trf_agg(trf_data, num_ranks)

def run_scalability(sz, is_variable):
    if (is_variable):
        sz = "variable"
    meta = {
        "ranks": [1, 2, 4, 6, 8],
        "sleep_times": [15],
        "prefetch_orders": [2],
        "num_prefetches": [0, 1, 2],
        "score_eviction": [0, 1],
    }
    
    data = {}
    for num_ranks in meta["ranks"]:
        data[num_ranks] = {}
        for s in meta["sleep_times"]:
            data[num_ranks][s] = {}
            for po in meta["prefetch_orders"]:
                for x, npf in enumerate(meta["num_prefetches"]):
                    for se in [meta["score_eviction"][x]]:
                        data[num_ranks][s][se] = {}
                        if (sz != "variable"):
                            size_in_mb = (1<<(sz-20))
                            filename = f"{results_folder}/res-uni-{num_ranks}-{size_in_mb}-{s}-{po}-{npf}-{se}.log" 
                        else:
                            filename = f"{results_folder}/res-variable-{num_ranks}-{s}-{po}-{npf}-{se}.log" 
                        print(f"Reading: {filename} ")
                        with open(filename) as f:
                            lines = f.readlines()
                            pf_time, next_prefetch, trf_agg = read_recs(num_ranks, lines, sz, po)                                  
                            data[num_ranks][s][se]["time"] = sum(pf_time)
    plot_scalability(meta, data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please supply folder containing all logs")
    
    results_folder = sys.argv[1]    
    is_variable = False
    is_scalability = False
    if len(sys.argv) > 2:
        is_variable = int(sys.argv[2])
    if len(sys.argv) > 3:
        is_scalability =int(sys.argv[3])
    print(f"Result folder: {results_folder}, is variable: {is_variable}, is scalability: {is_scalability}")
    if is_scalability:
        run_scalability(sizes[0], is_variable)