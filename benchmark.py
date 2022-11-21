class Benchmark:
    def __init__(self,mode):
        if mode == "macsim":
            self.benchmark_list= [
            #"backprop",
            "bfs",
            "cfd",
            "hotspot",
            "kmeans",
            #"lavaMD", C extern is used with template
            #"lud", #seg fault but we can get result
            #"nw",
            #"srad",
            "streamcluster",
            #"pathfinder",
            #"particlefilter",
            #"gaussian",
            "nn",
            #"heartwall",
            "hybridsort",  #need
            #"b+tree",NOOO
            #"dwt2d", NOOO
            #"hotspot3D", NOOO
            #"leukocyte", NOOO
            #"myocyte", NOOO
            ]
        elif mode == "igpu":
            self.benchmark_list= [
            "backprop",
            "bfs",
            "cfd",
            "hotspot",
            "kmeans",
            #"lavaMD", C extern is used with template
            "lud", #seg fault but we can get result
            "nw",
            "srad",
            #"streamcluster",
            "pathfinder",
            #"particlefilter",
            "gaussian",
            "nn",
            "heartwall",
            "hybridsort",  #need
            #"b+tree",NOOO
            #"dwt2d", NOOO
            #"hotspot3D", NOOO
            #"leukocyte", NOOO
            #"myocyte", NOOO
            ]
        else:
            print("ERR: select macsim or igpu")
            exit()

    def get_benchmark_list(self):
        return self.benchmark_list

    def get_benchmark_index(self,name):
        return self.benchmark_list.index(name)
