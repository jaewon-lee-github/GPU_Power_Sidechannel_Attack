#!/usr/bin/env python3
from array import array
from ctypes import sizeof
from functools import reduce
import re
import os
import glob
from pathlib import Path
import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='run power estimation')
parser.add_argument('--no-build', '-n', action='store_false', default=True)
parser.add_argument('--clean', '-c', action='store_true', default=False)
parser.add_argument('--max_inst', type=int, action='store', default=100000)
parser.add_argument('--freq-start', type=float, action='store', default=0.3)
parser.add_argument('--freq-interval', type=float, action='store', default=0.1)
parser.add_argument('--freq-max', type=float, action='store', default=2.0)
parser.add_argument('--time', type=int, action='store')
parser.add_argument('--cycle', type=int, action='store')
parser.add_argument('--inst', type=int, action='store')
args = parser.parse_args()

if args.time == False and args.inst == False and args.cycle == False:
    print("!!!!need to set --time, --inst, or --cycle!!!")
    exit()

"""
CAT 1 = ALU Operation
CAT 2 = L1_HIT_GPU
CAT 3 = TOTAL_DRAM_MERGE
CAT 4 = Total Instruction
total cycles
"""
re_INST = re.compile(r'PARENT_UOP_CORE_[0-9]+[0-9]*')
re_SEND = re.compile(r'OP_CAT_GED_SEND[CS]*[CS]*_CORE_[0-9]+[0-9]*')
# re_INST = re.compile(r'PARENT_UOP_CORE_0')
# re_SEND = re.compile(r'OP_CAT_GED_SEND[CS]*[CS]*_CORE_0')
re_L1_MISS = re.compile(r'L1_MISS_GPU')
re_DRAM = re.compile(r'TOTAL_DRAM')
re_CYCLE = re.compile(r'CYC_COUNT_CORE_0')

# {"file to search" : [[counter nubmer, Regular expression],...],...}
pattern_dict = {
    "inst.stat.out": [[0, re_INST], [1, re_SEND], ],
    "memory.stat.out": [[2, re_L1_MISS], [3, re_DRAM]],
    "general.stat.out": [[4, re_CYCLE]]  # 5, Frequency
}

now = datetime.datetime.now()
date = '%s%d' % (now.strftime("%b").lower(), now.day)
np.set_printoptions(suppress=True)

macsim_path = Path('/fast_data/jaewon/GPU_SCA/macsim')


def target_param(args):
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
    result_path = macsim_path / 'bin' / 'dvfs_test' / (str(interval) + suffix)
    num_sample = len(
        [i for i in Path(result_path / ("%2.1fGHz" % (args.freq_start))).glob('inst.stat.out.%s_*' % (suffix))])
    target_start = interval
    target_list = [i for i in range(
        target_start, target_start+interval*(num_sample-1)+1, interval)]
    return target_list, interval, target_start, num_sample, result_path, suffix

# select target here


target_list, interval, target_start, num_sample, result_path, suffix = target_param(
    args)

frq_list = [i for i in np.round(np.arange(
    args.freq_start, args.freq_max+args.freq_interval, args.freq_interval), decimals=1)]
stat_list = ["INST", "SEND", "L1_MISS", "DRAM", "CYCLE"]
type_list = ["ALU", "L1_MISS", "DRAM", "INST"]

# tg_inst = 1000
# inst_list = [tg_inst, tg_inst]
# 5 counters for each files
counters = np.zeros([len(frq_list), len(target_list), len(stat_list)])
counters_fixed = np.zeros([len(frq_list), len(target_list), len(type_list)])

counter_fname = 'counter_%d%s.npy' % (interval, suffix)
counter_fixed_fname = 'counter_fixed_%d%s.npy' % (interval, suffix)
counter_file = result_path / counter_fname
counter_fixed_file = result_path / counter_fixed_fname

if args.clean == True:
    os.system("rm %s %s" % (counter_fixed_file, counter_file))


if not counter_file.exists() or not counter_fixed_file.exists():
    for frq, i in enumerate(frq_list):
        new_dir = str(result_path)+"/"+str(i)+"GHz"
        os.chdir(new_dir)
        print("DIR: %s" % (new_dir))
        # counters[frq, :, -1] = i
        for t, j in enumerate(target_list):
            for k in pattern_dict.keys():
                for f in glob.glob('%s.%s_%d' % (k, suffix, j)):
                    with open(f) as tfile:
                        for line in tfile:
                            for d in pattern_dict[k]:
                                if (d[1].search(line)):
                                    counters[frq][t][d[0]] = counters[frq][t][d[0]] + \
                                        int(line.split()[1])
    counters_fixed[:, :, 0] = counters[:, :, 0] - counters[:, :, 1]
    counters_fixed[:, :, 1:3] = counters[:, :, 2:4]
    counters_fixed[:, :, 3] = counters[:, :, 0]
    os.chdir(str(result_path))
    with open(counter_fname, 'wb') as f:
        np.save(f, counters)
    with open(counter_fixed_fname, 'wb') as f:
        np.save(f, counters_fixed)
else:
    os.chdir(str(result_path))
    with open(counter_file, 'rb') as f:
        counters = np.load(f, allow_pickle=True)
    with open(counter_fixed_file, 'rb') as f:
        counters_fixed = np.load(f, allow_pickle=True)

# stat_list = ["INST", "SEND", "L1_MISS", "DRAM", "CYCLE" ]
with open("counters.csv", 'w') as f:
    np.savetxt(f, counters.reshape(-1, 5), delimiter=',', fmt="%10.1f")
with open("counter_fixed.csv", 'w') as f:
    np.savetxt(f, counters_fixed.reshape(-1, 4), delimiter=',', fmt="%10.1f")

# [From Hz, to Hz], scenario length should be same with num_sample
# scenario_list = [[1, 2], [2, 1]]
c_max = 11
c_start = 1
c_interval = 5
c_array = np.arange(c_start, c_max+1, c_interval).reshape((1, -1))
cm = [1, 1, 1, 1]


def x(n):
    return np.arange(c_start, c_max+1, c_interval).reshape((1, -1))*n


# type_list = ["ALU", "L1_MISS", "DRAM", "INST"]
# c_matrix = np.array(np.meshgrid(x(c_interval, c_max), x(
#     c_interval, c_max*2), x(c_interval, c_max*3), x(c_interval, c_max*4), x(c_interval, c_max*5)),).T
c_matrix = np.array(np.meshgrid(
    x(cm[0]), x(cm[1]), x(cm[2]), x(cm[3]))).T.reshape(-1, 4)
c_offset = [1000 for _ in range(num_sample)]
# c_matrix[:, [0, 1]] = c_matrix[:, [1, 0]]
# c_matrix[:, 4] = c_matrix[:, 4] *0.5


os.chdir(str(result_path))

# def pdiff_per_inst(frq_list, stat_list, type_list, inst_period, inst_list, counters, counters_fixed, scenario_list, c_matrix, est_power):
#     for scn_idx, scenario in enumerate(scenario_list):
#         for inst_idx, inst in enumerate(inst_list):
#             cur_freq = scenario[inst_idx]
#             result_cur = est_power(frq_list, stat_list, inst_list, counters,
#                                    counters_fixed, inst, c_matrix, cur_freq)
#             if inst_idx == 0:
#                 prev_freq = cur_freq
#                 result_prev = result_cur
#                 continue
#             print("\n################  Inst : %d - %d , Freq : %2.1f to %2.1f  ######################" %
#                   (inst-1000, inst, prev_freq, cur_freq))
#             print(type_list)
#             print(counters_fixed[frq_list.index(
#                 prev_freq), inst_list.index(inst-inst_period), :])
#             print(counters_fixed[frq_list.index(
#                 cur_freq), inst_list.index(inst), :])
#             power_diff = (result_cur-result_prev).reshape(-1)
#         # find the criteria
#         # margin = np.std(power_diff)/20
#             margin = 0
#             criteria = np.min(np.absolute(power_diff)) + margin  # Make some
#             print("Margin : %f, Crietria : %f" % (margin, criteria))

#             idx_power_diff = np.array(np.nonzero(
#                 np.absolute(power_diff) <= criteria)).reshape(-1)
#             print("num of elements within criteria : %d " %
#                   (len(idx_power_diff)))

#         # union the power diff and const matrix
#             res_matrix = np.concatenate(
#                 (c_matrix, power_diff.reshape((-1, 1))), axis=1)
#             file_name = "test_%d_%d_%d.out" % (prev_freq, cur_freq, inst)
#             with open(file_name, 'w', encoding="utf-8") as f:
#                 # np.savetxt(f, res_matrix,
#                 np.savetxt(f, res_matrix[idx_power_diff, :],
#                            delimiter=',', fmt='%10.5f',
#                            header="%10s,%10s,%10s,%10s,%10s,%10s" %
#                            ("c_ALU", "c_L1_MISS", "c_DRAM", "c_INST", "c_OFFSET", "DIFF"))

#             os.system("cat %s" % (file_name))
#             prev_freq = cur_freq
#             result_prev = result_cur


# pdiff_per_inst(frq_list, stat_list, type_list, inst_period, inst_list, counters, counters_fixed, scenario_list, c_matrix, est_power)

# init = 0
# # only for sample is
# for freq_index, freq in enumerate(frq_list):
#     if scenario[0] == scenario[1]:
#         continue
#     for freq_idx, freq in enumerate(scenario):
#         target = target_start
#         target_idx = 0
#         cur_freq = freq
#         cur_freq_idx = frq_list.index(freq)
#         result_cur = est_power(cur_freq_idx, stat_list, target_idx, counters, c_offset,
#                                counters_fixed, target, c_matrix, cur_freq)
#         if freq_idx == 0:
#             prev_freq = cur_freq
#             prev_freq_idx = cur_freq_idx
#             result_prev = result_cur
#             continue
#         print("\n################  %d - %d , Freq : %2.1f to %2.1f  ######################" %
#               (target-interval, target, prev_freq, cur_freq))
#         print(type_list)
#         print(counters_fixed[prev_freq_idx, 0, :])
#         print(counters_fixed[cur_freq_idx, target_idx, :])

#         power_diff = (result_cur-result_prev).reshape(-1)
#         # find the criteria
#         margin = np.std(power_diff)/20
#         # margin = 0
#         criteria = np.min(np.absolute(power_diff)) + margin  # Make some
#         print("Margin : %f, Crietria : %f" % (margin, criteria))

#         idx_power_diff = np.array(np.nonzero(
#             np.absolute(power_diff) <= criteria)).reshape(-1)
#         print("num of elements within criteria : %d " %
#               (len(idx_power_diff)))

#         m1 = np.resize(
#             [prev_freq, cur_freq], (len(c_matrix), 2))
#         m2 = np.resize(
#             [c_offset[prev_freq_idx], c_offset[cur_freq_idx]], (len(c_matrix), 2))
#         cur_res_matrix = np.concatenate(
#             (m1, c_matrix, m2, power_diff.reshape(-1, 1)), axis=1)
#         print(cur_res_matrix[idx_power_diff, :])

#         # union the power diff and const matrix
#         if init == 0:
#             final_res_matrix = cur_res_matrix[idx_power_diff, :]
#             init = 1
#         else:
#             final_res_matrix = np.concatenate(
#                 (final_res_matrix, cur_res_matrix[idx_power_diff, :]), axis=0)

#         prev_freq = cur_freq
#         prev_freq_idx = cur_freq_idx
#         result_prev = result_cur


"""
First objective: when we change frequency from 1GHz to 2GHz, or 2GHz -> 1GHz
what constant will remain the power same.
1. The counter can determine consumed energy.
    - If we divide the energy by time,which is cycle over Frequency, Then it is definitely power
2. Get Freq from Scenario_x[0]
3. From 1000 to n*1000, get the power number from the equation above
4. power equation
    power = (Summation(Const_n*(Event_n)*F^2)+c_offset_f)/(cycle/F)
          = (Summation(Const_n*(Event_n)*F^2)+c_offset_f)/(cycle/F)
          = (Summation(Const_n*(Event_n)*F^2)+c_offset_f)*(F/cycle)
    power Diff = power_cur - power_prev
5. search the index make the power difference minimal
"""

# def est_power(freq_idx, stat_list, target_idx, counters, c_offset, counters_fixed, inst, c_matrix, cur_freq):
#     result_cur = c_matrix * counters_fixed[freq_idx, target_idx, :]
#     result_cur = np.add.reduce(result_cur, -1, keepdims=True) * (cur_freq**2)
#     result_cur = result_cur + c_offset[freq_idx]
#     result_cur = result_cur / \
#         (counters[freq_idx, target_idx, stat_list.index("CYCLE")]) * cur_freq
#     return result_cur

# counters = np.zeros([len(frq_list), len(target_list), len(stat_list)])
# counters_fixed = np.zeros([len(frq_list), len(target_list), len(type_list)])
result = counters_fixed[:, :, np.newaxis, :] * c_matrix
result = np.add.reduce(result, -1, keepdims=True)
square_frq_list = np.array(frq_list) * np.array(frq_list)
result = result * square_frq_list[-1, np.newaxis, np.newaxis, np.newaxis]
result = result + np.array(c_offset)[-1, np.newaxis, np.newaxis, np.newaxis]

if suffix == "ns":
    exe_time = interval
elif suffix == "cycle":
    exe_time = interval/np.array(frq_list)  # Freq = Cycle/Time
elif suffix == "inst":
    temp_exe_time = counters[:, :, stat_list.index(
        "CYCLE")] / np.array(frq_list).reshape(-1,1)  # Freq = Cycle/Time
    exe_time = temp_exe_time[:,:,np.newaxis,np.newaxis]

result = result/exe_time*1000

const = 1
# stat_list = ["INST", "SEND", "L1_MISS", "DRAM", "CYCLE" ]
with open("result_%d.csv"%(const), 'w') as f:
    np.savetxt(f, result[:, :, const, 0], delimiter=',', fmt="%10.1f")
with open("const_%d.csv" % (const), 'w') as f:
    np.savetxt(f, c_matrix[const, :], delimiter=',', fmt="%10.1f")

min_mat = np.fmin.reduce(result, axis=0, keepdims=True)
max_mat = np.fmax.reduce(result, axis=0, keepdims=True)
min_min_mat = np.fmin.reduce(min_mat, axis=1, keepdims=True)
max_max_mat = np.fmax.reduce(max_mat, axis=1, keepdims=True)

# range_mat = np.concatenate((min_mat, max_mat), axis=-1,dtype=[('min', float),('max',float)])
range_mat = np.rec.fromarrays([min_mat, max_mat], names='min,max')
# range_mat = np.array([(5, 4), (3, 2)],dtype=[('min', int),('max',int)])
print(range_mat)
sort_mat = np.sort(range_mat, order='min', axis=1)
print(sort_mat)

dtype = [('name', 'S10'), ('height', float), ('age', int)]
values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
          ('Galahad', 1.7, 38)]
a = np.array(values, dtype=dtype)       # create a structured array
b = np.sort(a, order='height')
c = np.sort(a, order=['age', 'height'])


true_map1 = np.any(np.greater_equal(min_mat, max_max_mat), axis=1)
true_map2 = np.any(np.less_equal(max_mat, min_min_mat), axis=1)

# dtype = [('prev_f', float), ('cur_f', float), ('c_ALU', float), ('c_L1_MISS', float),
#          ('c_DRAM', float), ('c_INST', float), ('c_off_PF', float), ('c_off_CF', float), ('DIFF', float)]
# fmt = '%10.1f, %10.1f, %10f, %10f, %10f, %10f, %10f, %10f, %10.3f'
# final_res_matrix.dtype = dtype

# res = np.sort(final_res_matrix.reshape(-1), order='DIFF')

# file_name = "test_%d.out" % (target)
# with open(file_name, 'w', encoding="utf-8") as f:
#     # np.savetxt(f, res_matrix,
#     np.savetxt(f, res.reshape(-1),
#                delimiter=',', fmt=fmt,
#                header="%10s,%10s,%10s,%10s,%10s,%10s,%10s,%10s,%10s" %
#                ("Prev_F", "Cur_F", "c_ALU", "c_L1_MISS", "c_DRAM", "c_INST", "c_offset_Prev_F", "c_offset_Cur_F", "DIFF"))
# os.system("cat %s" % (file_name))
