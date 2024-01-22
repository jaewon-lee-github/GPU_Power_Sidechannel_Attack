import os
from pathlib import Path
from env import myEnv

home="/home/jaewon/work"
tango_path=home + "Tango/GPU"

class Benchmark:
    def __init__(self, suite):
        self.myEnv = myEnv()
        # if suite== "macsim":
        #     self.benchmark_dict = [
        #         # "backprop",
        #         "bfs",
        #         "cfd",
        #         "hotspot",
        #         "kmeans",
        #         # "lavaMD", C extern is used with template
        #         # "lud", #seg fault but we can get result
        #         # "nw",
        #         # "srad",
        #         "streamcluster",
        #         # "pathfinder",
        #         # "particlefilter",
        #         # "gaussian",
        #         "nn",
        #         # "heartwall",
        #         "hybridsort",  # need
        #         # "b+tree",NOOO
        #         #"dwt2d", NOOO
        #         #"hotspot3D", NOOO
        #         #"leukocyte", NOOO
        #         #"myocyte", NOOO
        #     ]
        # elif suite== "igpu":
        #     self.benchmark_list = [
        #         "backprop",
        #         "bfs",
        #         "cfd",
        #         "hotspot",
        #         "kmeans",
        #         # "lavaMD", C extern is used with template
        #         "lud",  # seg fault but we can get result
        #         "nw",
        #         "srad",
        #         # "streamcluster",
        #         "pathfinder",
        #         # "particlefilter",
        #         "gaussian",
        #         "nn",
        #         "heartwall",
        #         "hybridsort",  # need
        #         # "b+tree",NOOO
        #         #"dwt2d", NOOO
        #         #"hotspot3D", NOOO
        #         #"leukocyte", NOOO
        #         #"myocyte", NOOO
        #     ]
        # elif suite == "ptx":
        #     self.benchmark_list = [
        #         "ScalarProd",
        #         "SobolQRNG",
        #         "Histogram",
        #         "Dxtc",
        #         "sssp",
        #         "streamcluster",  # seg fault but we can get result
        #         "lud-64",
        #         "bfs-dtc",
        #         "MergeSort",
        #     ]
        if suite == "rodinia_cuda":
            self.base_dir = self.myEnv.rodinia_dir
            self.benchmark_dict = dict(
                [
                    ("backprop", ""),
                    ("cfd", ""),
                    ("gaussian", ""),
                    ("bfs", ""),
                    #  ("heartwall", ""),
                    ("hotspot", ""),
                    #  ("kmeans", ""),
                    ("lavaMD", ""),
                    ("lud", ""),
                    ("nn", ""),
                    ("nw", ""),
                    ("srad_v1", "srad"),
                    # ("srad_v2", "srad"),
                    ("streamcluster", ""),
                    ("particlefilter", ""),
                    ("pathfinder", ""),
                    # ("mummergpu", ""),
                    # ("hybridsort", ""),
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
        elif suite == "tango_cuda2":
            self.base_dir = myEnv.tango2_dir
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
        #elif suite == "ft_cuda":
        #    self.base_dir = Path("/fast_data/jaewon/FasterTransformer/build")
        #    self.benchmark_dict = dict(
        #        [
        #
        #       ]
        #)
        else:
            print("ERR: select macsim or igpu")
            exit()

    def get_benchmark_list(self):
        return self.benchmark_dict.keys()

    # def get_benchmark_index(self, name):
    #     return self.benchmark_list.(name)
