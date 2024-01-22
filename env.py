import os
from pathlib import Path

class myEnv:
    def __init__(self):
        self.root_dir=Path("/home/jaewon/work")
        self.tango_dir= self.root_dir / "Tango/GPU"
        self.tango2_dir= self.root_dir / "Tango2/GPU"
        self.rodinia_dir= self.root_dir /"/rodinia_3.1/cuda"
        self.nvbit_dir= self.root_dir / "NVBit_Power/tools/power/"
        self.nvbit_so= self.nvbit_dir / "power.so"
        self.result_dir=self.root_dir/ "GPU_Power_Sidechannel_Attack/results"
        print(self.tango_dir)




