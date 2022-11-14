import csv,sys,os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from benchmark import Benchmark
from pathlib import Path

benchmark = Benchmark()
benchmark_list = benchmark.get_benchmark_list()

max_len = 0
max_value = 0
if len(sys.argv) == 2:
    input_name = sys.argv[1]
else:
    input_name = "result/result"

target_dir = Path('./out')
target_dir.mkdir(parents=True,exist_ok=True)

target_prefix= input_name
org_padded_file = target_prefix + '_org_padded.csv'
padded_file = target_prefix + '_padded.csv'
report_file = target_prefix +'_report'

with open(input_name,'r',newline='') as input:
    csvreader = csv.reader(input)
    rows = list(csvreader)

cur_data_len = 0
# Find the maximum length of data and normalize the data
for idx, row in enumerate(rows):
    #print (row)
    if len(row) <= 2:
        del rows[idx][:]
        continue
    row = list(map(int, row))
    rows[idx][0]=benchmark.get_benchmark_index(int(row[0])) # change to text
    if row[1]== 0:
        rows[idx][1] = "DVFS"
    cur_data_len =len(row[2:])
    if max_len < cur_data_len:
        max_len = cur_data_len
    if max_value < max(row[2:]):
        max_value = max(row[2:])
        max_value_index = idx

print("max_len = {}, max_value = {}th:{} ".format(max_len, max_value_index, max_value))

#padding zeros
for idx, row in enumerate(rows):
    cur_data_len =len(row[2:])
    if max_len > cur_data_len and cur_data_len > 0 :
        rows[idx].extend(np.zeros((max_len-cur_data_len), dtype=float))
    elif max_len ==cur_data_len:
        print("max line = ",max_len)
    else:
        print("error!: max_len = {}, len(row)= {}th:{}".format(max_len, idx, cur_data_len))
        continue
    rows[idx][2:] = np.array(row[2:]).astype(float)

column = []
column.extend(['label'])
column.append('freq')
column.extend([i for i in range(max_len)])

df_org = pd.DataFrame(data = rows, columns = column)
print("save to ",org_padded_file)
df_org.to_csv(org_padded_file)
print("Dataframe org \n",df_org.head())
df_org_long = df_org.melt(id_vars =['label','freq'], var_name = "time", value_name ="power")
print("Dataframe long \n",df_org_long.head())
opg_png=target_prefix+'_org_power_graph.png'

# WARNING : sweep interval should be 200
freq_list = [f for f in range(300,1101,200)]
freq_list.append("DVFS")

my_rotation = 45
print("freq_list = ",freq_list)
fg = sns.relplot(x="time", y="power", hue="freq", #hue_order = freq_list,
        col="label", col_wrap=3,
        height=11, aspect=1.5, linewidth=1,
        kind="line", data=df_org_long)
for axes in fg.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=my_rotation)
print("save to ",opg_png)
plt.savefig(opg_png)
# Using as_index=False for groupby(). If not, the input column name will be index,
# which cannot be used as column next time
sumdf= df_org_long.groupby(['label','freq'],as_index=False).sum()
print(sumdf.head())

fg= sns.catplot(x='label', y='power',hue='freq',#hue_order = freq_list,
        kind='bar', palette="ch:.25", data=sumdf)
for axes in fg.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=my_rotation)
opg_sum_png=target_prefix+'_org_power_graph_sum.png'
print("save to ",opg_sum_png)
plt.savefig(opg_sum_png)

#normalization
for idx, row in enumerate(rows):
    norm_data = np.array(row[2:]).astype(int)
    norm = np.linalg.norm(norm_data)
    rows[idx][2:]= norm_data/norm

df = pd.DataFrame(data = rows, columns = column)
print("save to ",padded_file)
df.to_csv(padded_file)

df_long = df.melt(id_vars =["label",'freq'], var_name = "time", value_name ="power")

pg_png=target_prefix+'_power_graph.png'
mg_png=target_prefix+'_multi_graph.png'
fg=sns.relplot(data=df_long,x="time",y="power",hue="label",style="freq",#style_order = freq_list,
        kind="line")
for axes in fg.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=my_rotation)
print("save to ",pg_png)
plt.savefig(pg_png)

fg=sns.relplot(x="time", y="power",hue="freq",#hue_order=freq_list,
        col="label", col_wrap=5,
        height=7, aspect=.75, linewidth=0.6,
        kind="line", data=df_long)
for axes in fg.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=my_rotation)

print("save to ",mg_png)
plt.savefig(mg_png)

