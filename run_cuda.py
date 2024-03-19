#! /usr/bin/env python3
import getopt
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from benchmark import Benchmark
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from env import myEnv

myEnv = myEnv()
# custom directory definition
nvbit_so = myEnv.nvbit_so
result_dir = myEnv.result_dir
nvbit_dir = myEnv.nvbit_dir


def handling_options():
    options, remainder = getopt.getopt(
        sys.argv[1:],
        "i:s:t:d:cmf:orn:x:q:v",
        [
            "interval=",
            "suite=",
            "iteration=",
            "device=",
            "clean_make",
            "make",
            "freq_mode=",
            "tool_make",
            "norun",
            "min_freq=",
            "max_freq=",
            "step_freq=",
            "verbose",
        ],
    )

    interval = 100
    make = False
    tmake = True # FIXME
    clean = False
    iteration = 1
    device = 1
    run = True
    verbose = 0
    #suite = "tango_cuda"
    suite = "rodinia_cuda"
    min_freq = 500
    max_freq = 2000
    step_freq = 100
    freq_mode = 1  # 0: no change(Natural DVFS), 1: Random binning, 2: Fixed Freq

    for opt, arg in options:
        if opt in ("-i", "--interval"):
            interval = int(arg)
        elif opt in ("-t", "--iteration"):
            iteration = int(arg)
        elif opt in ("-s", "--suite"):
            suite = arg
        elif opt in ("-d", "--device"):
            device = int(arg)
        elif opt in ("-c", "--clean_make"):
            clean = True
            make = True
        elif opt in ("-m", "--make"):
            make = True
        elif opt in ("--tool_make"):
            tmake = True
        elif opt in ("-f", "--freq_mode"):
            freq_mode = int(arg)
        elif opt in ("--min_freq"):
            min_freq = int(arg)
        elif opt in ("--max_freq"):
            max_freq = int(arg)
        elif opt in ("--step_freq"):
            step_freq = int(arg)
        elif opt in ("-r", "--norun"):
            run = False
        elif opt in ("-v", "--verbose"):
            verbose = 1
    return (
        interval,
        suite,
        make,
        tmake,
        device,
        clean,
        iteration,
        run,
        freq_mode,
        min_freq,
        max_freq,
        step_freq,
        verbose,
    )


def get_outfile(
    suite_name, freq_mode, iteration, interval, min_freq, max_freq, freq_interval
):
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m%d%Y_%H%M%S")
    ofile_name = f"result_{suite_name}_{dt_string}_mode_{freq_mode}_x{iteration}_{interval}ms_{min_freq}MHz_{max_freq}MHz_{freq_interval}MHz.csv"
    # print("*** Output file =", ofile_name)
    return ofile_name


def run_benchmark_suite(options):
    (
        interval,
        suite,
        make,
        tmake,
        device,
        clean,
        iteration,
        run,
        freq_mode,
        min_freq,
        max_freq,
        step_freq,
        verbose,
    ) = options
    benchmark_list = benchmark.get_benchmark_list()
    cuda_path = os.environ.get("CONDA_PREFIX")
    print("=" * 10, "Environment Variables", "=" * 10)
    print("CUDA_PATH=", cuda_path)
    print("BENCH_PATH", benchmark.base_dir)
    print("=" * 40)

    if tmake == True:
        print("**** Make nvbit_tool")
        myEnv
        os.chdir(nvbit_dir)
        os.system("make clean")
        if verbose == 1:
            os.system("make VERBOSE=1 -j16")
        else:
            os.system("make -j16")
        os.chdir(benchmark.base_dir)

    os.environ["NOBANNER"] = "0"
    os.environ["MIN_FREQ"] = str(min_freq)
    os.environ["MAX_FREQ"] = str(max_freq)
    os.environ["STEP_FREQ"] = str(step_freq)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    for bm in benchmark_list:
        print("=" * 10, bm, "=" * 10)
        # print("**** DVFS reset**")
        # os.system("sudo nvidia-smi -rgc -rmc > /dev/null 2>&1")

        os.environ["BENCH_NAME"] = bm
        os.environ["FREQ_MODE"] = str(freq_mode)
        bm_dir = benchmark.base_dir / benchmark.benchmark_dict[bm] / bm
        os.chdir(bm_dir)
        if clean == True:
            print("\tClean " + str(bm_dir))
            os.system("make clean")
        if make == True:
            print("\tMake " + str(bm_dir))
            os.system("make")
        if run == True:
            print("\t******Run " + str(bm_dir) + "********")
            print("** DVFS reset")
            os.system("sudo nvidia-smi -rmc -rgc > /dev/null 2>&1")
            with open("./run", "r") as file:
                for line in file:
                    if line.strip().startswith("#") or not line.strip():
                        continue
                    else:
                        run_command = line.strip()
                        break
            try:
                run_command = f"sudo LD_PRELOAD={nvbit_so} CUDA_VISIBLE_DEVICES={device} NOBANNER=0 TOOL_VERBOSE={verbose} INTERVAL={interval} FREQ_MODE={freq_mode} BENCH_NAME={bm} PATH={cuda_path}/bin:$PATH MIN_FREQ={min_freq} MAX_FREQ={max_freq} STEP_FREQ={step_freq} {run_command}"
                # if verbose == 0:
                #     run_command = run_command + "> /dev/null 2>&1"
                # run_command = f"sudo LD_PRELOAD={nvbit_so} INTERVAL={interval} FREQ_MODE={freq_mode} BENCH_NAME={bm} PATH={cuda_path}/bin:$PATH {run_command}"
                print(run_command)
                retcode = subprocess.call(f"{run_command}", shell=True)
                if retcode < 0:
                    print("Child was terminated by signal", -retcode, file=sys.stderr)
                else:
                    print("Child returned", retcode, file=sys.stderr)
            except OSError as e:
                print("Execution failed:", e, file=sys.stderr)
            # os.system(run_command)


if __name__ == "__main__":
    result_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(result_dir)
    options = handling_options()
    (
        interval,
        suite,
        make,
        tmake,
        device,
        clean,
        iteration,
        run,
        freq_mode,
        min_freq,
        max_freq,
        step_freq,
        verbose,
    ) = options
    print(
        f"intervals={interval}, suite={suite}, make={make}, tmake={tmake}, device={device}, clean={clean}, iteration={iteration}, run={run}, freq_mode={freq_mode}, min_freq={min_freq}, max_freq={max_freq}, step_freq={step_freq}, verbose={verbose}"
    )

    benchmark = Benchmark(suite)
    # datetime object containing current date and time
    ofile_name = get_outfile(
        suite, freq_mode, iteration, interval, min_freq, max_freq, step_freq
    )

    # final result: original, cut to min length
    fin_wide_df = None
    threshold_sec = 4  # Change this to your desired threshold
    threshold = threshold_sec * 1000 / interval
    for i in tqdm(range(iteration)):
        acc_df = None
        run_benchmark_suite(options)
        os.chdir(benchmark.base_dir)
        # Already moved to benchmark directory
        # accumulate result
        for file in glob.iglob(
            f"./**/output_{freq_mode}_{min_freq}_{max_freq}_{step_freq}_*_{interval}ms.csv", recursive=True
        ):
            print(f"current file: {benchmark.base_dir}/{file}")
            df = pd.read_csv(file)
            if acc_df is None:
                acc_df = df
            else:
                acc_df = pd.concat([acc_df, df], ignore_index=True, axis=0)
            # os.remove(file)

        if acc_df is None:
            print("ERR: No result file")
            exit()

        # final result cut to min length
        # Step 1: Calculate the count of entries for the specified field
        value_counts = acc_df["Kernel"].value_counts()
        # Step 2: Create a boolean mask based on the condition
        mask = acc_df["Kernel"].map(value_counts) >= threshold
        filt_acc_df = acc_df[mask]
        # filt_acc_df= acc_df.groupby("Kernel").filter(lambda group: len(group) >= threshold)
        # Step 3: Use the boolean mask to filter the DataFrame

        acc_pvt_df = filt_acc_df.pivot(
            index="Kernel", columns="Timestamp", values="Power"
        )
        acc_pvt_df.reset_index(inplace=True)
        # print(acc_pvt_df)

        if fin_wide_df is None:
            fin_wide_df = acc_pvt_df
        else:
            fin_wide_df = pd.concat(
                [fin_wide_df, acc_pvt_df], ignore_index=True, axis=0
            )
        # print(fin_wide_df)

    # cut the dataframe to min length
    min = fin_wide_df.count(axis=1).min()
    ftd_acc_df_min = fin_wide_df.iloc[:, 0:min]
    print(ftd_acc_df_min)

    # for kernel in fin_value_counts.index:
    #     if ftd_acc_df_min is None:
    #         ftd_acc_df_min = ftd_acc_df[ftd_acc_df["Kernel"] == kernel].iloc[0:min]
    #     else:
    #         ftd_acc_df_min = pd.concat(
    #             [
    #                 ftd_acc_df_min,
    #                 ftd_acc_df[ftd_acc_df["Kernel"] == kernel].iloc[0:min],
    #             ],
    #             ignore_index=True,
    #             axis=0,
    #         )

    # df_wide.to_csv(ofile_name, index=True, mode="w")
    os.chdir(result_dir)
    fin_wide_df.to_csv(f"full_{ofile_name}", index=False, mode="w")
    ftd_acc_df_min.to_csv(ofile_name, index=False, mode="w")
    print(f"## store to csv {result_dir}/{ofile_name}")

    # output.write(uncore)dd
    print("**** DVFS reset")
    os.system("sudo nvidia-smi -rgc -rmc > /dev/null 2>&1")
    print("**** Create symbolic link to result")
    link = Path("result.csv")
    link.unlink(missing_ok=True)
    link.symlink_to(ofile_name)
