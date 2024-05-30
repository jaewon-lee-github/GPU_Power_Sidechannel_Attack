#! /usr/bin/env python3

import matplotlib
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # Statistical plotting
import os
import glob

# from scipy import stats
from scipy.stats import gmean
from pathlib import Path
from benchmark import Benchmark
import matplotlib.font_manager as fm
from env import myEnv

# custom directory definition
myEnv = myEnv()
result_dir = myEnv.result_dir


# bloomberg color code
def bb_color_pallete(num_cat):
    bb_color = {}
    bb_color["green"] = "#00BB96"
    bb_color["blue"] = "#0340D6"
    bb_color["orange"] = "#FE6663"
    bb_color["grey"] = "#636363"
    bb_color["black"] = "#000000"
    bb_color["light_blue"] = "#41C2FF"
    bb_color["light_grey"] = "#D3D3D3"

    if num_cat == 1:
        cp = [bb_color["black"]]
    elif num_cat == 2:
        cp = [bb_color["light_blue"], bb_color["black"]]
    elif num_cat == 3:
        cp = [bb_color["light_blue"], bb_color["green"], bb_color["black"]]
    elif num_cat == 4:
        # cp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        cp = [
            bb_color["light_blue"],
            bb_color["grey"],
            bb_color["green"],
            bb_color["black"],
        ]
    else:
        print(f"Not supported number of category({num_cat})")
        exit(1)

    return cp


def draw_bar_multigraph(input, output_file, column):
    # params for theme
    # FontProperties
    # family: A list of font names in decreasing order of priority. The items may include a generic font family name, either 'sans-serif', 'serif', 'cursive', 'fantasy', or 'monospace'. In that case, the actual font to be used will be looked up from the associated rcParam during the search process in findfont. Default: rcParams["font.family"] (default: ['sans-serif'])
    # ex)
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Calibri'
    # style: Either 'normal', 'italic' or 'oblique'. Default: rcParams["font.style"] (default: 'normal')
    # variant: Either 'normal' or 'small-caps'. Default: rcParams["font.variant"] (default: 'normal')
    # stretch: A numeric value in the range 0-1000 or one of 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded' or 'ultra-expanded'. Default: rcParams["font.stretch"] (default: 'normal')
    # weight: A numeric value in the range 0-1000 or one of 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'. Default: rcParams["font.weight"] (default: 'normal')
    # size: Either a relative value of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large' or an absolute font size, e.g., 10. Default: rcParams["font.size"] (default: 10.0)
    # math_fontfamily: The family of fonts used to render math text. Supported values are: 'dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans' and 'custom'. Default: rcParams["mathtext.fontset"] (default: 'dejavusans')

    ## Theme setting
    # Context: notebook”, “paper”, “talk”, and “poster”
    # Style: white, dark, whitegrid, darkgrid, ticks
    # Palette: deep, muted, bright, pastel, dark, colorblind
    # color : https://seaborn.pydata.org/tutorial/color_palettes.html
    # font family :  'sans-serif', 'serif', 'cursive', 'fantasy', or 'monospace'
    # How to install Calibri Font?
    ## sudo apt install fontforge
    ## sudo apt install cabextract
    ## wget https://gist.github.com/maxwelleite/10774746/raw/ttf-vista-fonts-installer.sh -q -O - | sudo bash
    ## fc-list | grep Calibri
    ## print(fm.findfont("Calibri"))
    ## print(fm.findSystemFonts(fontpaths=None, fontext="ttf"))

    sns.set_theme(
        "paper",
        "whitegrid",
        "Blues",
        font_scale=1,
        rc={"font.family": "Calibri", "axes.grid": False},
    )
    num_cat = len(input["FreqMode"].unique())
    # plt.rcParams["font.family"] = "Calibri"
    # print(plt.rcParams.keys())

    # sns.set_context('paper', font_scale=1.0)
    # sns.set_style('whitegrid')
    # cp = sns.color_palette("Paired", n_colors=num_cat)
    # cp = sns.color_palette("crest",n_colors=num_cat)
    # cp = sns.cubehelix_palette(n_colors=num_cat, start=.5, rot=-.75)
    # cp = sns.cubehelix_palette(n_colors=num_cat, start=.5, rot=-.5)
    # cp = sns.color_palette("Blues")
    # cp = sns.color_palette("viridis",n_colors=num_cat)
    # cp = sns.color_palette("hls",n_colors=num_cat)
    # cp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    cp = bb_color_pallete(num_cat)
    # cp = sns.color_palette(n_colors=num_cat)
    print("How many colors in palette?", len(cp))

    # sns.axes_style("whitegrid")
    # baseline_color = cp[0]
    # Hertzpatch_color= cp[1]

    # input data structure
    # "Kernel,Timestamp,Freq,Power,Freq_mode"
    # input.to_csv(result_dir / "temp.csv", index=False, mode="w")
    # input['Kernel'] = input['Kernel'].map(lambda x: x.split('_')[-1])
    # num_ker = len(input["Kernel"].unique())
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("_", "\n"))
    # sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")
    # sns.catplot(data=tips, x="day", y="total_bill", hue="weekend", kind="box")
    # sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")

    g = sns.catplot(
        data=input,
        kind="bar",
        x="Kernel",
        y=column,
        hue="FreqMode",
        hue_order=["base", "random", "HertzPatch"],
        # style="Freq_mode",
        palette=cp,
        # col_wrap=math.ceil(num_ker / 2),
        # col_wrap=10,
        height=8,
        aspect=2.0,
        linewidth=0.5,
        # facet_kws={"xlim": (0, 100)},
    )
    g.tick_params(
        axis="x",
        direction="out",
        length=5,
        width=1,
        colors="k",
        grid_color="k",
        bottom=True,
        grid_alpha=0.1,
        labelbottom=True,
        rotation=40,
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        labelsize="20",
    )
    g.tick_params(
        axis="y",
        direction="out",
        length=5,
        width=1,
        colors="k",
        left=True,
        grid_color="k",
        grid_alpha=0.7,
        labelsize="20",
    )
    # g = sns.FacetGrid(df, col="Kernel", col_wrap=4, height=2, hue="Freq_mode", palette=cp)
    # g.map(sns.lineplot, x="Timestamp", y="Power", linewidth =1.0, errorbar=None)
    # bbox_to_anchor reference: https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib/39806180#39806180
    g.set_titles(
        # col_template="{col_name}", row_template="{row_name}", size=12, fontweight="bold"
        col_template="{col_name}",
        size=12,
        fontweight="bold",
    )
    # g.set_axis_labels("Time(*100ms)", "Power (mW)")
    g.set_axis_labels("", "Normalized " + column)
    # g.set_xticklabels("")
    g._legend.set_title("FreqMode")
    # g._legend.texts[0].set_text("Base")
    # g._legend.texts[1].set_text("Random")
    # g._legend.texts[2].set_text("Fixed_2GHz")

    sns.move_legend(
        g,
        "lower center",
        # bbox_to_anchor=(0.5, 0.97),
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=True,
        fontsize=12,
    )
    sns.despine(
        top=False, right=False, left=False, bottom=False, offset=None, trim=False
    )

    g.savefig(str(output_file) + ".pdf", format="pdf", bbox_inches="tight", dpi=600)
    print("output file " + str(output_file) + ".pdf generated")

    return g


# argument input : pandas dataframe longform
def draw_multigraph(input, output_file, column):
    # params for theme
    # FontProperties
    # family: A list of font names in decreasing order of priority. The items may include a generic font family name, either 'sans-serif', 'serif', 'cursive', 'fantasy', or 'monospace'. In that case, the actual font to be used will be looked up from the associated rcParam during the search process in findfont. Default: rcParams["font.family"] (default: ['sans-serif'])
    # ex)
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Calibri'
    # style: Either 'normal', 'italic' or 'oblique'. Default: rcParams["font.style"] (default: 'normal')
    # variant: Either 'normal' or 'small-caps'. Default: rcParams["font.variant"] (default: 'normal')
    # stretch: A numeric value in the range 0-1000 or one of 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded' or 'ultra-expanded'. Default: rcParams["font.stretch"] (default: 'normal')
    # weight: A numeric value in the range 0-1000 or one of 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'. Default: rcParams["font.weight"] (default: 'normal')
    # size: Either a relative value of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large' or an absolute font size, e.g., 10. Default: rcParams["font.size"] (default: 10.0)
    # math_fontfamily: The family of fonts used to render math text. Supported values are: 'dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans' and 'custom'. Default: rcParams["mathtext.fontset"] (default: 'dejavusans')

    ## Theme setting
    # Context: notebook”, “paper”, “talk”, and “poster”
    # Style: white, dark, whitegrid, darkgrid, ticks
    # Palette: deep, muted, bright, pastel, dark, colorblind
    # color : https://seaborn.pydata.org/tutorial/color_palettes.html
    # font family :  'sans-serif', 'serif', 'cursive', 'fantasy', or 'monospace'
    # How to install Calibri Font?
    ## sudo apt install fontforge
    ## sudo apt install cabextract
    ## wget https://gist.github.com/maxwelleite/10774746/raw/ttf-vista-fonts-installer.sh -q -O - | sudo bash
    ## fc-list | grep Calibri
    ## print(fm.findfont("Calibri"))
    ## print(fm.findSystemFonts(fontpaths=None, fontext="ttf"))

    sns.set_theme(
        "paper",
        "whitegrid",
        "Blues",
        rc={"font.family": "Calibri", "axes.grid": False},
    )
    num_cat = len(input["FreqMode"].unique())
    # plt.rcParams["font.family"] = "Calibri"
    # print(plt.rcParams.keys())

    # sns.set_context('paper', font_scale=1.0)
    # sns.set_style('whitegrid')
    # cp = sns.color_palette("Paired", n_colors=num_cat)
    # cp = sns.color_palette("crest",n_colors=num_cat)
    # cp = sns.cubehelix_palette(n_colors=num_cat, start=.5, rot=-.75)
    # cp = sns.cubehelix_palette(n_colors=num_cat, start=.5, rot=-.5)
    # cp = sns.color_palette("Blues")
    # cp = sns.color_palette("viridis",n_colors=num_cat)
    # cp = sns.color_palette("hls",n_colors=num_cat)

    # cp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    cp = bb_color_pallete(num_cat)

    # sns.axes_style("whitegrid")
    # baseline_color = cp[0]
    # Hertzpatch_color= cp[1]

    # input data structure
    # "Kernel,Timestamp,Freq,Power,Freq_mode"
    # input.to_csv(result_dir / "temp.csv", index=False, mode="w")
    # input['Kernel'] = input['Kernel'].map(lambda x: x.split('_')[-1])
    # num_ker = len(input["Kernel"].unique())
    # input["Kernel"] = input["Kernel"].map(lambda x: x.replace("_", "\n"))
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("_", "\n"))
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("Execute", ""))
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("execute", ""))

    g = sns.relplot(
        data=input,
        kind="line",
        x="Timestamp",
        y=column,
        hue="FreqMode",
        hue_order=["base", "random", "HertzPatch"],
        col="Kernel",
        # style="Freq_mode",
        palette=cp,
        # col_wrap=math.ceil(num_ker / 2),
        col_wrap=3,
        height=2,
        aspect=1.0,
        linewidth=0.5,
        # facet_kws={"xlim": (0, 100)},
    )
    g.tick_params(
        axis="x",
        direction="out",
        length=5,
        width=1,
        colors="k",
        grid_color="k",
        bottom=False,
        grid_alpha=0.1,
        labelbottom=False,
        labelsize=15,
    )
    g.tick_params(
        axis="y",
        direction="out",
        length=5,
        width=1,
        colors="k",
        left=True,
        grid_color="k",
        grid_alpha=1.0,
        labelsize=15,
    )
    g.set(xlim=(0, 100))
    # g = sns.FacetGrid(df, col="Kernel", col_wrap=4, height=2, hue="Freq_mode", palette=cp)
    # g.map(sns.lineplot, x="Timestamp", y="Power", linewidth =1.0, errorbar=None)
    # bbox_to_anchor reference: https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib/39806180#39806180
    g.set_titles(
        col_template="{col_name}", row_template="{row_name}", size=12, fontweight="bold"
    )
    # g.set_axis_labels("Time(*100ms)", "Power (mW)")

    if column == "Power":
        y_axis_label= "Power (mW)"
    elif column == "Freq":
        y_axis_label= "Frequency (MHz)"
    g.set_axis_labels("Time", y_axis_label)
    # g.set_xticklabels("")
    g._legend.set_title("FreqMode")
    # g._legend.texts[0].set_text("Base")
    # g._legend.texts[1].set_text("Random")
    # g._legend.texts[2].set_text("Fixed_2GHz")

    sns.move_legend(
        g,
        "lower center",
        # bbox_to_anchor=(0.5, 0.97),
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=True,
        fontsize=12,
    )
    sns.despine(
        top=False, right=False, left=False, bottom=False, offset=None, trim=False
    )

    g.savefig(str(output_file) + ".pdf", format="pdf", bbox_inches="tight", dpi=600)
    print("output file " + str(output_file) + ".pdf generated")

    return g


if __name__ == "__main__":
    benchmark = Benchmark(myEnv.benchmark_name)
    # freq_mode 0 = Natural, 1 = Random DVFS
    # load orginal dataframe
    interval = 100

    # df = pd.read_csv(f"{result_dir}/acc_df_{freq_mode}_{interval}ms.csv")
    data_dir = result_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cur_dir = os.getcwd()
    os.chdir(data_dir)
    print("cwd: ", os.getcwd())

    out_dir = myEnv.figure_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = [
        "long_result_dev0_tango_cuda_05192024_202343_mode_0_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv",
        "long_result_dev1_tango_cuda_05192024_202341_mode_2_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv",
        "long_result_dev0_tango_cuda_05202024_040556_mode_3_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv",
    ]
    long_df = None
    # long_df's columns : Iteration,Kernel,Timestamp,Freq,FreqMode,BinPolicy,Power
    for file in files:
        if long_df is None:
            long_df = pd.read_csv(
                file,
            )
        else:
            df = pd.read_csv(file)
            long_df = pd.concat([long_df, df], axis=0, ignore_index=True)

    def titling(x):
        if x == 0:
            return "base"
        elif x == 1:
            return "fixed"
        elif x == 2:
            return "random"
        elif x == 3:
            return "HertzPatch"

    long_df["FreqMode"] = long_df["FreqMode"].map(titling)
    long_df["Power"] = long_df["Power"] / 10

    # print(long_df)
    g_max_df = long_df.groupby(["FreqMode", "Kernel", "Iteration"]).max().reset_index()
    g_mean_df = (
        long_df.groupby(["FreqMode", "Kernel", "Iteration"]).mean().reset_index()
    )
    g_mean_df["Timestamp"] = g_max_df["Timestamp"]

    g_mean_df["Performance"] = 1 / (g_mean_df["Timestamp"].astype(float))
    g_mean_df["EDP"] = (
        g_mean_df["Power"] * g_mean_df["Timestamp"] * g_mean_df["Timestamp"]
    )
    # print(g_mean_df)

    # g_mean_df["ED^2P"] = g_mean_df["EDP"] * g_mean_df["Timestamp"]
    def normalized(group, targets):
        if (group["FreqMode"] == "base").any():
            for tgt in targets:
                val = float(group[tgt].loc[group["FreqMode"] == "base"].values[0])
                group[tgt] = group[tgt].div(val)
        else:
            print(group)
            print("No base category found in this group.")
        return group

    # Group by 'Store' and apply custom function
    target_list = ["Power", "Performance", "EDP","Freq"]
    g_norm_df = (
        g_mean_df.groupby(["Kernel", "Iteration"])
        .apply(normalized, targets=target_list)
        .reset_index(drop=True)
    )

    # print(g_norm_df.groupby(["FreqMode","Iteration"]).apply(gmean).reset_index())
    # for c in columns_to_calculate:
    #     g_norm_df[c] = g_norm_df[c].astype(float)
    # grouped = (
    #     g_norm_df.groupby(["FreqMode", "Iteration"])
    #     .apply(lambda x: gmean(x, axis=0))
    #     .reset_index()
    # )

    # for tgt in target_list:
    #     draw_bar_multigraph(g_norm_df, out_dir / f"bar_graph_{tgt}", tgt)

    # modes = ["base", "random", "HertzPatch"]
    # for tgt in modes:
    #     out_file = out_dir / f"line_graph_power_{tgt}"
    #     new_df = long_df.loc[long_df["FreqMode"] == tgt]
    #     draw_multigraph(new_df, out_file)
    # out_file = out_dir / "line_graph_All"
    # draw_multigraph(long_df, out_file)
    
    # modes = ["base", "random", "HertzPatch"]
    # for tgt in modes:
    #     out_file = out_dir / f"line_graph_freq_{tgt}"
    #     new_df = long_df.loc[long_df["FreqMode"] == tgt]
    #     draw_multigraph(new_df, out_file, "Freq")
    out_file = out_dir / "line_graph_freq_All"
    draw_multigraph(long_df, out_file,"Freq")

    # # Set the font size and style for x-axis label and tick labels
    # x_label = x_axis.get_label()
    # x_label.set_fontsize(9)  # Adjust the font size as needed
    # x_ticklabels = x_axis.get_ticklabels()
    # for label in x_ticklabels:
    #     label.set_fontsize(10)  # Adjust the font size for tick labels as needed

    # # Set the font size and style for y-axis label and tick labels
    # y_label = y_axis.get_label()
    # y_label.set_fontsize(12)  # Adjust the font size as needed
    # y_ticklabels = y_axis.get_ticklabels()
    # for label in y_ticklabels:
    #     label.set_fontsize(10)

    # # Set the font properties for the legend
    # for text in legend.texts:
    #     print(text.get_text())
    #     text.set_font_properties(font_prop)
    # ax_column.set_xlabel("", fontsize=9)
    # ax_column.set_title(workload, fontsize=13)

    # ax_column.set_ylabel("", fontsize=9)
    # ax_column.get_legend().remove()

    # # Graph

    # return g
    # for row, ax_row in enumerate(axes):
    #     for column, ax_column in enumerate(ax_row):
    #         idx = row * 2 + column
    #         workload = workloads[idx]
    #         sns.lineplot(
    #             data=org_df[workload],
    #             ax=ax_column,
    #             label="Baseline",
    #             color=baseline_color,
    #             linewidth=1.0,
    #         )
    #         sns.lineplot(
    #             data=dvfs_df[workload],
    #             ax=ax_column,
    #             label="HertzPatch",
    #             color=Hertzpatch_color,
    #             linewidth=1.0,
    #         )

    #         # thickness=1.0
    #         # for p in ['left', 'right', 'top', 'bottom']:
    #         #   plt.gca().spines[p].set_linewidth(thickness)
    #         #   plt.gca().spines[p].set_color('black')
    #         # # ax_column.set_yticks("")
    #         # ax_column.
    #         # ax_column.set_yticks([-1,200,400,1000],fontsize=10)
    #         # ax_column.xticks(fontsize=9)
    #         # ax_column.yticks(fontsize=9)
    #         ax_column.set_xlabel("", fontsize=9)
    #         # ax_column.set_ylabel('Power (W)',fontsize=9)
    #         ax_column.set_title(workload, fontsize=13)

    #         ax_column.set_ylabel("", fontsize=9)
    #         ax_column.get_legend().remove()
    #         # ax_column.

    # handles, labels = axes[-1][0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     bbox_to_anchor=(-1.5, 1.05),
    #     ncol=2,
    #     fontsize=12,
    # )
    # fig.text(-1.5, 0.02, "Time (* 10K Cycle)", ha="center", fontsize=15)
    # fig.text(-1, 0.5, "Power (W)", va="center", rotation="vertical", fontsize=15)
    # fig.tight_layout()

    # output_file = "Powertrace_Result"
    # output_file = output_dir / output_file
    # fig.savefig(output_file + ".pdf", format="pdf", bbox_inches="tight", dpi=600)


# def collect_data(benchmark, interval):
#     acc_df = None
#     cur_dir = os.getcwd()
#     os.chdir(benchmark.base_dir)
#     print(os.getcwd())
#     for file in glob.iglob(f"./**/output_*_*.csv", recursive=True):
#         # format : Kernel,Timestamp,Freq,Power
#         print(f"current file: {file}")
#         df = pd.read_csv(file)
#         if acc_df is None:
#             acc_df = df
#         else:
#             acc_df = pd.concat([acc_df, df], ignore_index=True, axis=0)
#         # os.remove(file)
#     if acc_df is None:
#         print("ERR: No result file")
#         exit()

#     # add Freq_mode column Now done in myNvml.cu
#     # new_column = acc_df.columns.tolist() + ["Freq_mode"]
#     # acc_df = acc_df.reindex(columns=new_column, fill_value=freq_mode)
#     os.chdir(cur_dir)
#     return acc_df


# def collect_data_freq_mode(benchmark, interval):
#     res_df = collect_data(benchmark, interval)

#     threshold_sec = 4  # Change this to your desired threshold
#     # Count the number of timestamps of each kernel
#     # if it is below threshold then remove it with mask
#     threshold = threshold_sec * 1000 / interval
#     value_counts = res_df["Kernel"].value_counts()
#     mask = res_df["Kernel"].map(value_counts) >= threshold
#     filt_acc_df = res_df[mask]
#     # filt_acc_df= acc_df.groupby("Kernel").filter(lambda group: len(group) >= threshold)
#     # Step 3: Use the boolean mask to filter the DataFrame

#     filt_wide_df = filt_acc_df.pivot(
#         index=["Kernel", "Freq_mode"], columns="Timestamp", values="Power"
#     )
#     filt_wide_df.reset_index(inplace=True)
#     # filt_acc_df.to_csv(f"{result_dir}/acc_df_{freq_mode}_{interval}ms.csv", index=False, mode="w")
#     # filt_wide_df.to_csv(f"{result_dir}/wide_df_{freq_mode}_{interval}ms.csv", index=False, mode="w")
#     return filt_acc_df, filt_wide_df
