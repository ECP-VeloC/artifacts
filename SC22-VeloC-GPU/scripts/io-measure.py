#!/bin/bash python3
import sys

def process_time_agg(a, r):
    num_ckpts = len(a)//r
    t = [0]*num_ckpts
    for c in range(num_ckpts):
        for i in range(r):
            t[c] +=a[i*num_ckpts+c]
        t[c] = t[c]/num_ranks
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

def readFile(file_input, num_ranks):
    start_index = -6
    with open(file_input) as f:
        lines = f.readlines()
        temp = []
        trf_data = []
        for r in range(num_ranks):
            temp.extend([int(x) for x in lines[start_index-num_ranks-r].split(",")[:-1]])
            trf_data.append(convert_to_dict(str(lines[start_index-r])))
        return process_time_agg(temp, num_ranks), process_trf_agg(trf_data, num_ranks)

if __name__ == "__main__":
    input_folder = sys.argv[1]
    is_rtm = False
    if (len(sys.argv) > 2):
        is_rtm = True
    host_caches = [32]
    gpu_caches = [4] 
    sleep_times=[10, 15]
    num_ranks = 8
    npfs = [0, 1, 2]
    pos = [0, 1, 2]
    res = []
    for g in gpu_caches:
        for h in host_caches:
            for s in sleep_times:
                mini = 1e20
                maxi = 0
                total_speedup = 0
                for po in pos: 
                    pf_name = "Same"
                    if (po == 1):
                        pf_name = "Reverse"
                    elif (po == 2):
                        pf_name = "Random"
                    for npf in npfs:
                        f_sc = f"{input_folder}/uniform-res-{g}G{h}H/res-uni-{num_ranks}-128-{s}-{po}-{npf}-0-1.log"
                        f_fifo = f"{input_folder}/uniform-res-{g}G{h}H/res-uni-{num_ranks}-128-{s}-{po}-{npf}-0-0.log"
                        if is_rtm:
                            f_sc = f"{input_folder}/rtm-res-{g}G{h}H/res-rtm-{num_ranks}-{s}-{po}-{npf}-0-1.log"
                            f_fifo = f"{input_folder}/rtm-res-{g}G{h}H/res-rtm-{num_ranks}-{s}-{po}-{npf}-0-0.log"
                        sc_vals, sc_trf = readFile(f_sc, num_ranks)
                        fifo_vals, fifo_trf = readFile(f_fifo, num_ranks)
                        mini = min(mini, min(sum(fifo_vals), sum(sc_vals)))
                        maxi = max(maxi, max(sum(fifo_vals), sum(sc_vals)))
                        speedup = round(sum(fifo_vals)/sum(sc_vals), 3)
                            round(sum(fifo_vals), 3), 
                            round(sum(sc_vals), 3), 
                            speedup
                        ))
                        total_speedup += speedup
                        f_to_h = 0 if fifo_trf["FtoH"] == 0 or sc_trf["FtoH"] == 0 else fifo_trf["FtoH"]/sc_trf["FtoH"]
                        h_to_d = 0 if fifo_trf["HtoD"] == 0 or sc_trf["HtoD"] == 0 else fifo_trf["HtoD"]/sc_trf["HtoD"]
                        print ("{:<25} {:<20} {:<20} {:<8} {:<8} {:<8}, FIFO-FtoH {:<8} SC-FtoH {:<8} | FIFO-HtoD {:<8} SC-HtoD {:<8}".format(f"{g}G-{h}H-{npf}NPF-{po}PO-{s}ms:", 
                                round(sum(fifo_vals), 3), 
                                round(sum(sc_vals), 3), 
                                speedup,
                                round(f_to_h, 3),
                                round(h_to_d, 3),
                                (fifo_trf["FtoH"] >> 20),
                                (sc_trf["FtoH"] >> 20),
                                (fifo_trf["HtoD"] >> 20),
                                (sc_trf["HtoD"] >> 20),
                            ))                        

