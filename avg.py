from env import myEnv
import os
import pandas as pd

myEnv = myEnv()
root_dir = myEnv.root_dir
result_dir = myEnv.result_dir
window_size = 20
cur_dir = os.getcwd()
os.chdir(result_dir)
print("cwd: ", os.getcwd())
file = "full_result_tango_cuda_05172024_175639_mode_2_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv"
out_window= file.split(".")[0] + f"_sliding_windows_{window_size}.csv"
df = pd.read_csv(file, index_col=0)
df_T = df.transpose()
df_T.rolling(window=window_size).mean().iloc[20:,:].transpose().to_csv(out_window)

# print(out_mean)
# out_mean= file.split(".")[0] + "_multi_traces_mean.csv"
# df.groupby('Kernel').mean().to_csv(out_mean)