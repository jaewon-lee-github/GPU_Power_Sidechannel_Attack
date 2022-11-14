import csv,sys,os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


input=sys.argv[1]


print("read csv")
df = pd.read_csv(input)
print(df.head())
print("melt data")
df_long = df.melt(id_vars =["label"], var_name = "time", value_name ="power")
print(df['label'].value_counts())

plt.figure(figsize = (12,7))
sns.relplot(data=df_long,x="time",y="power",hue="label",kind="line")
pg_png='power_graph.png'
mg_png='multi_graph.png'
print("save to ",pg_png)
plt.savefig(pg_png)

#plt.figure(figsize = (12,12))
sns.relplot(x="time", y="power",
        col="label", col_wrap=4,
        height=3, aspect=.75, linewidth=2.5,
        kind="line", data=df_long)
print("save to ",mg_png)
plt.savefig(mg_png)


exit(1)
g = sns.FacetGrid(df_long, col="label", col_wrap=2, height=2 )

print("map data")
g.map(sns.lineplot, "time","power")
print("save figure ")
