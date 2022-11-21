#! /usr/bin/env python3
import csv
import fileinput
import getopt
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import sleep

from tqdm import tqdm
from benchmark import Benchmark

# DVFS when freq is zero
def set_freq(freq):
    if freq != 0:
        print("input_freq= {}".format(freq))
        print("\t** Frequency will be locked to ",freq)
        os.system("sudo intel_gpu_frequency --set {}".format(freq))
        os.system("sudo intel_gpu_frequency --set {}".format(freq))
        print("*** Frequency info")
        os.system("sudo intel_gpu_frequency".format(freq))
    else:
        print("Set frequency to default (Min:300, Max:1100)")
        os.system("sudo intel_gpu_frequency -d")
        os.system("sudo intel_gpu_frequency")

# default interval is always 1
def set_filename(dt_string,input_interval,sweep,freq):
    output_file_name = "result_"+dt_string
    if freq == None :
       output_file_name = output_file_name + "_DVFS_"
    elif sweep == True:
       output_file_name = output_file_name + "_sweep_"+str(freq) + "MHz_"
    else:
       output_file_name = output_file_name + "_"+str(freq)+"MHz_"

    output_file_name = output_file_name +  str(input_interval)+"ms"
    print("*** Output file =", output_file_name)
    return output_file_name


if __name__ == '__main__':
# datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%y_%H%M%S")
    print("1. Current date and time =", dt_string)
    #rapl_dir = Path('./opencl') / "lib"
    result_dir = Path('./result')
    result_dir.mkdir(parents=True, exist_ok=True)
    rodinia_ocl_dir = Path('/home/jaewon/work/rodinia_3.1/opencl')
    rapl_dir = rodinia_ocl_dir / "lib"

    options, remainder = getopt.getopt(sys.argv[1:], 'i:t:cf:smlr',
        ['interval=',
        'iteration=',
        'clean_make',
        'freq=',
        'sweep',
        'make',
        'librapl',
        'run',
        ])
    print ('2. OPTIONS   :', options)

    interval = 1
    input_interval = None
    input_freq = None
    make = False
    clean = False
    iteration = 1
    rapl_make = False
    run = True
    sweep = False

    for opt, arg in options:
        if opt in ('-i','--interval'):
            input_interval= int(arg)
        elif opt in ('-t','--iteration'):
            iteration= int(arg)
        elif opt in ('-c','--clean_make'):
            clean= True
            make= True
        elif opt in ('-f','--freq'):
            input_freq= int(arg)
        elif opt in ('-s','--sweep'):
            sweep= True
        elif opt in ('-m','--make'):
            make= True
        elif opt in ('-l','--librapl'):
            rapl_make= True
            clean = True
            make= True
        elif opt in ('-r','--norun'):
            run = False


    if input_interval != None:
        interval = input_interval # else interval is 1 as it defined

    os.chdir("result")
    cwd = os.getcwd()

    if rapl_make == True:
        print("*** Compiling rapl")
        os.chdir(rapl_dir)
        os.system("make clean && make")
        os.chdir(cwd)


    # input_freq != None -> default DVFS
    # sweep == True => Sweep distance is freq (freq = 200, Sweep 300,500,700,..,1100)
    # freq == sweep distance or target frequency
    freq_list = []
    if input_freq != None:
        if sweep == True:
            freq_list.extend(list(range(300, 1101,input_freq)))
            freq_list.append(0)
        else:
            freq_list.extend(input_freq)
    else:
        freq_list.extend([0])
    print(freq_list)

    # output_file_name = result_dir / set_filename(dt_string,interval,sweep,input_freq)
    output_file_name = set_filename(dt_string,interval,sweep,input_freq)
    output= open(output_file_name, "w")
    uncore = ""
    benchmark = Benchmark("igpu")

    for cur_freq in freq_list:
        set_freq(cur_freq)
        benchmark_list = benchmark.get_benchmark_list()
        for bm in benchmark_list:
            print ("====",bm,",",benchmark.get_benchmark_index(bm),"====")
            bm_dir =rodinia_ocl_dir / bm
            os.chdir(bm_dir)
            if input_interval != None:
                print("\tChange interval to",input_interval,"is not implemented")
                #for file in glob.iglob('*.c*'):
                    #print (file)
                    #with open(file, "r") as sources:
                    #    for line in lines:
                    #        if re.search(r'rapl_interval =', line):
                    #            print(line)
                    #            found = 1;
                    #            #sources.write(re.sub(r'rapl_interval[ ]*=[ ]*', 'deb', line))
                exit (1)
            if clean == True:
                print("\tClean " + str(bm_dir))
                os.system("make clean")
            if make == True:
                print("\tMake " + str(bm_dir))
                os.system("make")
            if run == True:
                print("\t******Run " + str(bm_dir)+"********")
                for i in tqdm(range(iteration)):
                    os.system("sudo ./run > /dev/null 2>&1")
                    #os.system("sudo ./run")
                    # names of the results are output_[benchmark]_[interval]ms.
                    for file in glob.iglob(r'output_'+bm+'_'+str(interval)+'ms'):
                        with open(file, newline='') as csvfile:
                            dictReader = csv.DictReader(csvfile)
                            # Use wideformat for saving storage
                            line = ""
                            for row in dictReader:
                                #print(row)
                                if line == "":
                                    line =  row['uncore']
                                else:
                                    line = line + ','+row['uncore']
                            uncore = uncore + str(benchmark.get_benchmark_index(bm)) + "," + str(cur_freq)+"," + line + '\n'
            # print(uncore)
            os.chdir(cwd)
    output.write(uncore)
    output.close()
    link = Path("result")
    link.unlink(missing_ok=True)
    link.symlink_to(output_file_name)
    # In the sweep sequence, it will set back DVFS at last.
    if input_freq != None and sweep == False:
        print("Set frequency to default (Min:300, Max:1100)")
        os.system("sudo intel_gpu_frequency -d")
        os.system("sudo intel_gpu_frequency")

