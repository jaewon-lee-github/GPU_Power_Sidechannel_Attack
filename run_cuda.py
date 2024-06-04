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

myEnv = myEnv("cuda")
# custom directory definition
nvbit_so = myEnv.nvbit_so
result_dir = myEnv.result_dir
nvbit_dir = myEnv.nvbit_dir


def handling_options():
    options, remainder = getopt.getopt(
        sys.argv[1:],
        "i:s:t:d:cmf:orn:x:q:v",
        [
            "sampling_interval=",
            "reset_interval=",
            "suite=",
            "iteration=",
            "device=",
            "clean_make",
            "make",
            "freq_mode=",
            "bin_policy=",
            "tool_make",
            "norun",
            "min_freq=",
            "max_freq=",
            "step_freq=",
            "default",
            "verbose",
        ],
    )

    # param for per-kernel meausre mode
    bin_policy = 10  # 10 => measure per kernel instead of thread.
    sampling_interval = 50

    # param for per-thread measure mode
    # bin_policy = 0  # 10 => measure per kernel instead of thread.
    # sampling_interval = 100
    reset_interval = 2000

    make = False
    tmake_clean = True
    # tmake_clean = True
    clean = False
    iteration = 1
    device = 0
    run = True
    verbose = 0
    suite = myEnv.suite_name
    min_freq = 400
    max_freq = 2000
    step_freq = 400
    # 0: Org DVFS, 1: Fixed, 2: Random, 3: Adaptive
    freq_mode = 0

    for opt, arg in options:
        if opt in ("-i", "--sampling_interval"):
            sampling_interval = int(arg)
        elif opt in ("--reset_interval"):
            reset_interval = int(arg)
        elif opt in ("-t", "--iteration"):
            iteration = int(arg)
        elif opt in ("-d", "--device"):
            device = int(arg)
        elif opt in ("-c", "--clean_make"):
            clean = True
            make = True
        elif opt in ("-m", "--make"):
            make = True
        elif opt in ("--tool_make"):
            tmake_clean = True
        elif opt in ("-f", "--freq_mode"):
            freq_mode = int(arg)
        elif opt in ("--min_freq"):
            min_freq = int(arg)
        elif opt in ("--max_freq"):
            max_freq = int(arg)
        elif opt in ("--step_freq"):
            step_freq = int(arg)
        elif opt in ("--bin_policy"):
            bin_policy = int(arg)
        elif opt in ("-r", "--norun"):
            run = False
        elif opt in ("--default"):
            print(
                f"sampling_interval={sampling_interval}, reset_interval={reset_interval} suite={suite}, make={make}, tmake={tmake_clean}, device={device}, clean={clean}, iteration={iteration}, run={run}, freq_mode={freq_mode}, bin_policy={bin_policy},min_freq={min_freq}, max_freq={max_freq}, step_freq={step_freq}, verbose={verbose}"
            )
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = 1
        else:
            sys.exit()
    return (
        sampling_interval,
        reset_interval,
        suite,
        make,
        tmake_clean,
        device,
        clean,
        iteration,
        run,
        freq_mode,
        bin_policy,
        min_freq,
        max_freq,
        step_freq,
        verbose,
    )


def get_outfile(
    suite_name,
    device,
    freq_mode,
    bin_policy,
    iteration,
    sampling_interval,
    reset_interval,
    min_freq,
    max_freq,
    freq_interval,
):
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m%d%Y_%H%M%S")
    ofile_name = f"result_dev{device}_{suite_name}_{dt_string}_mode_{freq_mode}_{bin_policy}_x{iteration}_{sampling_interval}ms_{reset_interval}ms_{min_freq}MHz_{max_freq}MHz_{freq_interval}MHz.csv"
    # print("*** Output file =", ofile_name)
    return ofile_name


def run_benchmark_suite(options):
    (
        sampling_interval,
        reset_interval,
        suite,
        make,
        tmake_clean,
        device,
        clean,
        iteration,
        run,
        freq_mode,
        bin_policy,
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

    for bm in benchmark_list:
        print("=" * 10, bm, "=" * 10)
        # print("**** DVFS reset**")
        # os.system(f"sudo nvidia-smi -i {device} -rgc > /dev/null 2>&1")

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
            os.system(f"sudo nvidia-smi -i {device} -rgc > /dev/null 2>&1")
            with open("./run", "r") as file:
                for line in file:
                    if line.strip().startswith("#") or not line.strip():
                        continue
                    else:
                        run_command = line.strip()
                        break
            try:
                run_command = f"sudo LD_PRELOAD={nvbit_so} CUDA_VISIBLE_DEVICES={device} NOBANNER=0 TOOL_VERBOSE={verbose} SAMPLING_INTERVAL={sampling_interval} RESET_INTERVAL={reset_interval} FREQ_MODE={freq_mode} BIN_POLICY={bin_policy} BENCH_NAME={bm} PATH={cuda_path}/bin:$PATH MIN_FREQ={min_freq} MAX_FREQ={max_freq} STEP_FREQ={step_freq} {run_command}"
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


def accumulate_df(final, cur):
    if final is None:
        final = cur
    else:
        final = pd.concat([final, cur], ignore_index=True, axis=0)

    return final


if __name__ == "__main__":
    result_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(result_dir)
    options = handling_options()
    (
        sampling_interval,
        reset_interval,
        suite,
        make,
        tmake_clean,
        device,
        clean,
        iteration,
        run,
        freq_mode,
        bin_policy,
        min_freq,
        max_freq,
        step_freq,
        verbose,
    ) = options
    print(
        f"sampling_interval={sampling_interval}, reset_interval={reset_interval} suite={suite}, make={make}, tmake={tmake_clean}, device={device}, clean={clean}, iteration={iteration}, run={run}, freq_mode={freq_mode}, bin_policy={bin_policy},min_freq={min_freq}, max_freq={max_freq}, step_freq={step_freq}, verbose={verbose}"
    )

    benchmark = Benchmark(myEnv)
    # datetime object containing current date and time
    ofile_name = get_outfile(
        suite,
        device,
        freq_mode,
        bin_policy,
        iteration,
        sampling_interval,
        reset_interval,
        min_freq,
        max_freq,
        step_freq,
    )

    # final result: original, cut to min length
    fin_wide_df = None
    fin_long_df = None
    print("**** Make nvbit_tool")
    os.chdir(nvbit_dir)
    if tmake_clean == True:
        os.system("make clean")
    if verbose == 1:
        os.system("make VERBOSE=1 -j16")
    else:
        os.system("make -j16")
    os.chdir(benchmark.base_dir)

    for i in tqdm(range(iteration)):
        acc_df = None
        run_benchmark_suite(options)
        os.chdir(benchmark.base_dir)
        # Already moved to benchmark directory
        # accumulate result
        for file in glob.iglob(
            f"./**/output_{device}_{freq_mode}_{bin_policy}_{min_freq}_{max_freq}_{step_freq}_*_{sampling_interval}_{reset_interval}.csv",
            recursive=True,
        ):
            print(f"current file: {benchmark.base_dir}/{file}")
            df = pd.read_csv(file)
            df["Platform"] = myEnv.platform_name
            df["Device"] = myEnv.device_name
            df["Iteration"] = i
            acc_df = accumulate_df(acc_df, df)
            # os.remove(file)

        if acc_df is None:
            print("ERR: No result file")
            exit()

        if bin_policy != 10:  # When kernel mode
            Kernels = [
                ["AlexNet", "execute3DconvolutionCuda"],
                ["AlexNet", "executeFCLayer"],
                ["AlexNet", ",executelrnNormCuda_split"],
                ["CifarNet", "ExecuteFirstLayer"],
                ["CifarNet", "ExecuteSecondLayer"],
                ["CifarNet", "ExecuteThirdLayer"],
                ["LSTM", "ExecuteLSTM"],
                ["ResNet", "execute3DconvolutionCuda_split"],
                ["ResNet", "executeFirstLayerCUDA"],
                ["SqueezeNet", "ExecuteFirstLayer"],
                ["SqueezeNet", "ExecuteTenthLayer"],
                ["SqueezeNet", "Executefire4expand3x3"],
                ["SqueezeNet", "Executefire5expand3x3"],
            ]
            mask = {}
            for kernel in Kernels:
                cur_mask = (acc_df["Benchmark"] == kernel[0]) & (
                    acc_df["Kernel"] == kernel[1]
                )
                if len(mask):
                    mask = mask | cur_mask
                else:
                    mask = cur_mask
            filt_acc_df = acc_df[mask]
        else:
            filt_acc_df = acc_df

        fin_long_df = accumulate_df(fin_long_df, filt_acc_df)
        print(fin_long_df)

        acc_pvt_df = filt_acc_df.pivot(
            index=["Benchmark", "Kernel"], columns="Timestamp", values="Power"
        )
        acc_pvt_df.reset_index(inplace=True)
        fin_wide_df = accumulate_df(fin_wide_df, acc_pvt_df)

    # cut the dataframe to min length
    min = fin_wide_df.count(axis=1).min()
    ftd_acc_df_min = fin_wide_df.iloc[:, 0:min]

    # df_wide.to_csv(ofile_name, index=True, mode="w")
    os.chdir(result_dir)
    fin_long_df.to_csv(f"long_{ofile_name}", index=False, mode="w")
    fin_wide_df.to_csv(f"full_{ofile_name}", index=False, mode="w")
    ftd_acc_df_min.to_csv(ofile_name, index=False, mode="w")
    print(f"## store long form to csv {result_dir}/long_{ofile_name}")
    print(f"## store wide form to csv {result_dir}/full_{ofile_name}")

    # output.write(uncore)dd
    print("**** DVFS reset")
    os.system(f"sudo nvidia-smi -i {device} -rgc > /dev/null 2>&1")
    print("**** Create symbolic link to result")
    long_link = Path("long_result.csv")
    long_link.unlink(missing_ok=True)
    long_link.symlink_to("long_" + ofile_name)

    wide_link = Path("wide_result.csv")
    wide_link.unlink(missing_ok=True)
    wide_link.symlink_to("full_" + ofile_name)
