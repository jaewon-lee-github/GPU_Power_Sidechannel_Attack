import os
from pathlib import Path


class Benchmark:
    def __init__(self, e):
        self.suite = e.suite_name
        if self.suite == "rodinia_cuda":
            self.base_dir = e.rodinia_cuda_dir
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
                    # ("lud", ""),
                    ("nn", ""),
                    ("nw", ""),
                    ("srad", "srad_v1"),
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
        elif self.suite == "tango_cuda":
            self.base_dir = e.tango_dir
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
        elif self.suite == "rodinia_ocl":
            self.base_dir = e.rodinia_ocl_dir
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
                    # ("lud", "ocl"),
                    ("nn", ""),
                    ("nw", ""),
                    ("srad", ""),  # different structure from cuda version
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
