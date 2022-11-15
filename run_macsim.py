#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import datetime
from time import time
import numpy as np
import argparse

from joblib import Parallel, delayed

macsim_path = Path('/fast_data/jaewon/GPU_SCA/macsim')
bin_path = macsim_path / 'bin'

def run_program(frq, cmd, result_path, args):
    if frq != "DVFS":
        frq_str = '%2.1fGHz'%(float(frq)/10.0)
        new_dir = result_path / frq_str
    else:
        new_dir = result_path / "DVFS"

    if not new_dir.is_dir():
        os.system("mkdir -p " + str(new_dir))
        os.chdir(str(new_dir))
        os.system("ln -sf "+str(macsim_path)+"/.opt_build/macsim macsim")
        os.system("ln -sf "+str(macsim_path)+"/bin/params.in params.in")
        os.system("ln -sf "+str(macsim_path) +
                  "/bin/trace_file_list trace_file_list")
    else:
        os.chdir(str(new_dir))
    print("CUR_DIR =%s" % (str(new_dir)))
    if args.no_clean is False:
        print("** Clean the output files in the DIR **")
        os.system('rm -f *.out*')
    if frq == "DVFS":
        cmd = cmd + "--clock_gpu=%s " % ("15")
        cmd = cmd + "--enable_dyfr=1 "
    else:
        cmd = cmd + "--clock_gpu=%s " % (frq)
        cmd = cmd + "--enable_dyfr=0 "
    cmd = cmd + "--dyfr_gpu_power_tgt=%s" % ("5000")
    os.system(cmd)
    return 1


def main():
    parser = argparse.ArgumentParser(description='run macsim simulation in parallel')
    parser.add_argument('--no-build', action='store_true', default=False)
    parser.add_argument('--no-clean', '-c', action='store_true', default=False)
    parser.add_argument('--max_inst', type=int, action='store', default=10000000)
    parser.add_argument('--time', type=int, action='store')
    parser.add_argument('--cycle', type=int, action='store')
    parser.add_argument('--inst', type=int, action='store')
    parser.add_argument('--freq-start', type=float, action='store', default=3)
    parser.add_argument('--freq-interval', type=float, action='store', default=1)
    parser.add_argument('--freq-max', type=float, action='store', default=22)
    args = parser.parse_args()

    frq_list = ["%d" % i for i in np.arange(args.freq_start,args.freq_max+args.freq_interval,args.freq_interval)]
    frq_list.append("DVFS")

    cmd = "./macsim --max_insts=%d " % (args.max_inst)
    if args.time is not None:
        cmd = cmd + "--stat_time_ns_interval=%d " % (args.time)
        interval = args.time
        suffix = "ns"
    elif args.cycle is not None:
        cmd = cmd + "--stat_cycle_interval=%d " % (args.cycle)
        interval = args.cycle
        suffix = "cycle"
    elif args.inst is not None:
        cmd = cmd + "--stat_inst_interval=%d " % (args.inst)
        interval = args.inst
        suffix = "inst"
    else:
        print("Error: set --time or --inst, or --cycle")
        exit()

    result_path = bin_path / 'dvfs_test' / 'cmp_frq'/ ( suffix + str(interval))
    if args.no_build == False:
        os.chdir(str(bin_path))
    #    os.system("make clean")
        os.system("./build.py -j16")
    # with open(bin_path/"app.txt", 'w') as f:
    #     f.write()

    results = Parallel(n_jobs=-1, verbose=10)(delayed(run_program)(frq, cmd, result_path, args)
                                              for frq in frq_list)
    #run_program("1.0", cmd, result_path, args)


if __name__ == '__main__':
    main()
