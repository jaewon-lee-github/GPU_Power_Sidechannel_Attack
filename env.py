import os
from pathlib import Path


class myEnv:
    def __init__(self, target = "cuda"):

        self._target = target
        # Common Directories
        # self.root_dir = Path("/fast_data/jaewon/GPU_SCA/power_patch")
        self.root_dir = Path("/home/jaewon/hertzpatch/")
        self.result_dir = self.root_dir / "GPU_Power_Sidechannel_Attack/results"
        self.figure_dir = self.result_dir / "../figures"
        if target == "cuda":
            # CUDA
            self.platform_name = "NVIDIA GPU"
            self.device_name = "RTX2080"
            # self.device_name = "GTX 1660"
            # self.device_name = "Titan Xp"
            self.benchmark_name = "rodinia_cuda"
            # self.benchmark_name = "tango_cuda"
            self.tango_dir = self.root_dir / "Tango/GPU"
            self.rodinia_cuda_dir = self.root_dir / "rodinia_3.1/cuda"
            self.nvbit_dir = self.root_dir / "NVBit_Power/tools/power/"
            self.nvbit_so = self.nvbit_dir / "power.so"
        elif target == "ocl":
            # OpenCL 
            self.platform_name = "Intel GPU"
            self.device_name = "UHD770"
            self.benchmark_name = "rodinia_ocl"
            self.rodinia_ocl_dir = self.root_dir / "rodinia_3.1/opencl"
            self.rapl_dir = self.rodinia_ocl_dir/ "lib"


        self.print_all()

    def print_all(self):
        # Common Information
        print("Platform_name =", self.platform_name)
        print("device_name =", self.device_name)
        print("benchmark_name =", self.benchmark_name)

        # Common Directories
        print("root_dir =", self.root_dir)
        print("result_dir=", self.result_dir)
        print("figure_dir=", self.figure_dir)

        if self._target == "cuda":
            # cuda benchmarks
            print("tango_dir=", self.tango_dir)
            print("rodinia_cuda_dir =", self.rodinia_cuda_dir)
            print("nvbit_dir=", self.nvbit_dir)
            print("nvbit_so=", self.nvbit_so)
        elif self._target == "ocl":
            # opencl benchmarks
            print("rodinia_ocl_dir =", self.rodinia_ocl_dir)
            print("rapl_dir=",self.rapl_dir)
