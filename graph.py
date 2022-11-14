import csv,sys,os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if len(sys.argv) == 2:
    input = sys.argv[1]
    output = "graph.png"
else:
    input = ""
    output = "graph.png"
#with open(input,'r',newline ='') as input:
#    csvreader = csv.reader(input)
#    rows = list(csvreader)

sns.set_theme(style="whitegrid")
df= pd.read_csv(input, header=None,index_col=-1)
plt.figure(figsize = (12,7))
sns.relplot(data=df,kind="line")
plt.savefig('graph.png')

