import os
from pathlib import Path


class myEnv:
    def __init__(self):
        self.root_dir = Path("/fast_data/jaewon/GPU_SCA/power_patch")
        self.tango_dir = self.root_dir / "Tango/GPU"
        self.tango2_dir = self.root_dir / "Tango2/GPU"
        self.rodinia_dir = self.root_dir / "rodinia_3.1/cuda"
        self.nvbit_dir = self.root_dir / "NVBit_Power/tools/power/"
        self.nvbit_so = self.nvbit_dir / "power.so"
        self.result_dir = self.root_dir / "GPU_Power_Sidechannel_Attack/results"

    def print_all(self):
        print("root_dir =",self.root_dir)
        print("tango_dir=",self.tango_dir )
        print("tango2_dir =",self.tango2_dir )
        print("rodinia_dir =",self.rodinia_dir )
        print("nvbit_dir=",self.nvbit_dir )
        print("nvbit_so=",self.nvbit_so)
        print("result_dir=",self.result_dir )

