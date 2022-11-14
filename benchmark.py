class Benchmark():
    def __init__(self):
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

    def get_benchmark_list(self):
        return self.benchmark_list

    def get_benchmark_index(self,name):
        return self.benchmark_list.index(name)