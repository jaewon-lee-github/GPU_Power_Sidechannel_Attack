import os, sys

servers = [
    # [
    #     "jaewon@fiat.cc.gt.atl.ga.us",
    #     "/home/jaewon/hertzpatch/GPU_Power_Sidechannel_Attack/results",
    #     "Intel_UHD770",
    # ],  # server, path, name pui
    # [
    #     "jaewon@amaranth.cc.gt.atl.ga.us",
    #     "/home/jaewon/work/GPU_Power_Sidechannel_Attack/results",
    #     "Nvidia_GTX_1660_Ti",
    # ],  # 23
    [
        "jlee3639@nio.cc.gatech.edu",
        "/home/jlee3639/hertzpatch/GPU_Power_Sidechannel_Attack/results",
        "Nvidia_GTX_3060",
    ],  # gt
]
directory = "gathered_results"
files = ["long_result.csv", "wide_result.csv"]
os.system(f"mkdir -p {directory}")
for server in servers:
    for file in files:
        os.system(f"scp {server[0]}:{server[1]}/{file}  {directory}/{server[2]}_{file}")
