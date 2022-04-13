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
sleep_times["rtm"]=[10, 15]
prefetch_orders = [0, 1, 2]
num_prefetches = [0, 1, 2]
score_eviction=[0, 1]
pname = ["Same", "Reverse", "Random"]
gpu_cache_size = 4
host_cache_size = 32

def save(x, y_all, leg, xlabel, ylabel, file):
    fig, ax = plt.subplots(figsize=(8, 4))
    ls = [":", "-.", "--", "-"]
    marks = ["o", "s", "x", "v", "^", "*"]
    for i, y in enumerate(y_all):
        markers_on=[x for x in range(0+i*len(y)//36, len(y), len(y)//6)]
        ax.plot(x, y, label=leg[i], linestyle=ls[i%len(ls)], markevery=markers_on, marker=marks[i])
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(frameon= 0, prop={'size': 14, 'weight': 'bold'})
    fig.tight_layout() 
    fig.savefig(f"{file}", format="pdf", bbox_inches='tight')

def plot_graph(sz, s, results_folder, data, all_times, next_prefetched):
    fig, axs = plt.subplots(3, 3, figsize=(30, 12))
    x = np.arange(len(data[0][0]))
    size = 0
    if sz != "rtm" and sz > 0:
        size = 1<<(sz-20)    
    Path(f"{results_folder}/plots/").mkdir(parents=True, exist_ok=True)
    outfile = f"{results_folder}/plots/rtm-{s}_ms_interval"   
    if size > 0:
        outfile = f"{results_folder}/plots/{size}_MB-{s}_ms_interval"
    print(f"In plot graph for {sz, s, results_folder}")
    xlab = "Restore operation number"
    ylab = ["Cumulative restore wait (ms)", "Individual restore wait (ms)", "Number of next prefetches completed"]
    for po in prefetch_orders:
        print(f"Plotting for size {sz}, sleep: {s}, prefetch order: {po}")
        ind = 0
        labs = []
        for npf in num_prefetches:
            for se in score_eviction:
                npf_str = "0 PF"
                if npf == 1:
                    npf_str = "1 PF"
                elif npf == 2:
                    npf_str = "* PF"  
                se_str = "FIFO  "
                if se == 1:
                    se_str = "Score"
                w_str = f"PO: {po}, NPF: {npf}, SE: {se}, ind: {ind}: {data[po][ind][-1]}"
                speed_str = f"{npf_str}, {se_str} ({round(data[po][ind][-1]/data[po][len(data[po])-1][-1], 2)})"
                labs.append(speed_str)
                axs[0][po].plot(x, data[po][ind], label=speed_str, )
                axs[1][po].plot(x, all_times[po][ind], label=speed_str)
                axs[2][po].plot(x, next_prefetched[po][ind], label=speed_str)                
                ind += 1
        save(x, data[po], labs, xlab, ylab[0], f'{outfile}_PF_{pname[po]}_cumulative.pdf')
        save(x, all_times[po], labs, xlab, ylab[1], f'{outfile}_PF_{pname[po]}_individual.pdf')
        save(x, next_prefetched[po], labs, xlab, ylab[2], f'{outfile}_PF_{pname[po]}_next_pf.pdf')
        axs[0][po].set_title(f"Restore order: {pname[po]}")
        for i in range(3):
            axs[i][po].set_ylabel(ylab[i])
            axs[i][po].set_xlabel(xlab)
            axs[i][po].legend()
        
    if size > 0:
        fig.suptitle(f"Ckpt size: {size} MB, Num snapshots: {len(data[0][0])}, GPU cache: {gpu_cache_size} GB, Host cache: {host_cache_size} GB, interval {s} ms")
    else:
        fig.suptitle(f"RTM traces, Num snapshots: {len(data[0][0])}, GPU cache: {gpu_cache_size} GB, Host cache: {host_cache_size} GB, interval {s} ms")
    fig.tight_layout()    
    print(f"Saved at: {outfile}.png")
    fig.savefig(f"{outfile}.png", bbox_inches='tight')

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

def read_size(sz):
    # Read cache sizes
    get_host_gpu_cache(results_folder)
    
    for s in sleep_times[sz]:
        for num_ranks in ranks:
            agg_sum_all = []
            per_pf_time_all = []
            next_prefetched_all = []
            for po in prefetch_orders:                
                agg_sum = []
                per_pf_time = []
                next_prefetched = []
                for npf in num_prefetches:
                    for se in score_eviction:
                        if (sz != "rtm"):
                            size_in_mb = (1<<(sz-20))
                            filename = f"{results_folder}/res-uni-{num_ranks}-{size_in_mb}-{s}-{po}-{npf}-{se}.log" 
                        else:
                            filename = f"{results_folder}/res-rtm-{num_ranks}-{s}-{po}-{npf}-{se}.log" 
                        print("Running file: ", filename, len(agg_sum))
                        with open(filename) as f:
                            lines = f.readlines()
                            pf_time, next_prefetch, _ = read_recs(num_ranks, lines, sz, po)
                            per_pf_time.append(pf_time)
                            agg_sum.append(np.cumsum(pf_time))
                            next_prefetched.append(next_prefetch)
                        print(f"{filename} at index {len(per_pf_time)-1}")
                        
                agg_sum_all.append(agg_sum)
                per_pf_time_all.append(per_pf_time)  
                next_prefetched_all.append(next_prefetched)
            plot_graph(sz, s, results_folder, agg_sum_all, per_pf_time_all, next_prefetched_all)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please supply folder containing all logs")
    
    results_folder = sys.argv[1]    
    is_rtm = False
    if len(sys.argv) > 2:
        is_rtm = int(sys.argv[2])
    print(f"Result folder: {results_folder}, is RTM: {is_rtm}")
    elif is_rtm:
        read_size("rtm")
    else:
        for sz in sizes:
            read_size(sz)