import os
from pathlib import Path
from env import myEnv


class Benchmark:
    def __init__(self, suite):
        self.myEnv = myEnv()
        self.suite = suite
        if suite == "rodinia_cuda":
            self.base_dir = self.myEnv.rodinia_cuda_dir
            self.benchmark_dict = dict(
                [
                    ("backprop", ""),
                    ("cfd", ""),
                    ("gaussian", ""),  #  ./gaussian -s 3500
                    ("bfs", ""),
                    ("heartwall", ""),
                    # ("hotspot", ""),
                    # ("kmeans", ""),
                    # ("lavaMD", ""),
                    ("lud", ""),
                    ("nn", ""),
                    ("nw", ""),
                    ("srad_v1", "srad"),
                    # ("srad_v2", "srad"),
                    ("streamcluster", ""),
                    ("particlefilter", ""),
                    ("pathfinder", ""),
                    # ("mummergpu", ""),
                    ("hybridsort", ""),
                    # ("dwt2d", ""),
                    # ("leukocyte",""),
                ]
            )
        elif suite == "tango_cuda":
            self.base_dir = self.myEnv.tango_dir
            self.benchmark_dict = dict(
                [
                    ("AlexNet", ""),
                    ("CifarNet", ""),
                    ("GRU", ""),
                    ("LSTM", ""),
                    ("ResNet", ""),
                    ("SqueezeNet", ""),
                ]
            )
        if suite == "rodinia_ocl":
            self.base_dir = self.myEnv.rodinia_ocl_dir
            self.benchmark_dict = dict(
                [
                    ("backprop", ""),
                    ("cfd", ""),
                    ("gaussian", ""),  #  ./gaussian -s 3500
                    ("bfs", ""),
                    ("heartwall", ""),
                    # ("hotspot", ""),
                    # ("kmeans", ""),
                    # ("lavaMD", ""),
                    ("lud", ""),
                    ("nn", ""),
                    ("nw", ""),
                    ("srad", ""), # different structure from cuda version
                    # ("srad_v2", "srad"),
                    ("streamcluster", ""),
                    ("particlefilter", ""),
                    ("pathfinder", ""),
                    # ("mummergpu", ""),
                    ("hybridsort", ""),
                    # ("dwt2d", ""),
                    # ("leukocyte",""),
                ]
            )
        else:
            print("ERR: select macsim or igpu")
            exit()

    def get_benchmark_list(self):
        return self.benchmark_dict.keys()

    # def get_benchmark_index(self, name):
    #     return self.benchmark_list.(name)
