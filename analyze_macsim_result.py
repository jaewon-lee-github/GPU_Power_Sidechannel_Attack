#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import datetime
from time import time
import numpy as np
import argparse
from benchmark import Benchmark

from joblib import Parallel, delayed

bm = Benchmark("macsim")

macsim_path = Path('/fast_data/jaewon/GPU_SCA/macsim')
bin_path = macsim_path / 'bin'

def main():
    parser = argparse.ArgumentParser(description='run macsim simulation in parallel')
    parser.add_argument('--benchmark', type=ascii, action='store',default='ALL')
    parser.add_argument('--time', type=int, action='store')
    parser.add_argument('--cycle', type=int, action='store')
    parser.add_argument('--inst', type=int, action='store')
    parser.add_argument('--freq-start', type=float, action='store', default=3)
    parser.add_argument('--freq-interval', type=float, action='store', default=1)
    parser.add_argument('--freq-max', type=float, action='store', default=21)
    args = parser.parse_args()

    frq_list = ["%d" % i for i in np.arange(args.freq_start,args.freq_max+1,args.freq_interval)]
    frq_list.append("DVFS")

    if args.time is not None:
        interval = args.time
        suffix = "ns"
    elif args.cycle is not None:
        interval = args.cycle
        suffix = "cycle"
    elif args.inst is not None:
        interval = args.inst
        suffix = "inst"
    else:
        print("Error: set --time or --inst, or --cycle")
        exit()

    result_path = bin_path / 'dvfs_test' / 'cmp_frq'/ benchmark / ( suffix + str(interval))
    #f = open(result_path/"out_analysis.csv","rt")

    os.system("rm %s/out_analysis.csv"%(str(result_path)))
    for frq in frq_list:
        if frq != "DVFS":
            frq_str = '%2.1fGHz'%(float(frq)/10.0)
            new_dir = result_path / frq_str
        else:
            new_dir = result_path / "DVFS"

        if not new_dir.is_dir():
            print(new_dir," is not existed.")
            exit()

        os.chdir(str(new_dir))
        print(new_dir)
        os.system("cat power_trace.csv >> ../out_analysis.csv")
        os.system("echo >> ../out_analysis.csv")


    return 1

if __name__ == '__main__':
    main()
