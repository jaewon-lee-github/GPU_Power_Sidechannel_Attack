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


def draw_bar_multigraph(input, output_file, x_axis, y_axis, group, column):
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
    # input["Kernel"] = input["Kernel"].map(lambda x: x.replace("_", "\n"))
    # sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")
    # sns.catplot(data=tips, x="day", y="total_bill", hue="weekend", kind="box")
    # sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")

    g = sns.catplot(
        data=input,
        kind="bar",
        x=x_axis,
        y=y_axis,
        hue=group,
        # hue_order=["base", "random", "HertzPatch"],
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
    g.set_axis_labels("", "Normalized " + y_axis)
    # g.set_xticklabels("")
    g._legend.set_title(group)
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
def draw_line_multigraph(input, output_file, x_axis, y_axis, group, column):
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
    # input["Kernel"] = input["Kernel"].map(lambda x: x.replace("_", "\n"))
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("Execute", ""))
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("execute", ""))

    g = sns.relplot(
        data=input,
        kind="line",
        x=x_axis,
        y=y_axis,
        hue=group,
        # hue_order=["base", "random", "HertzPatch"],
        col=column,
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

    if y_axis == "Power":
        y_axis_label = "Power (mW)"
    elif y_axis == "Freq":
        y_axis_label = "Frequency (MHz)"
    g.set_axis_labels("Time", y_axis_label)
    # g.set_xticklabels("")
    g._legend.set_title(group)
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
    benchmark = Benchmark(myEnv)
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
        "long_result_dev0_rodinia_cuda_06032024_234844_mode_0_10_x1_50ms_2000ms_400MHz_2000MHz_400MHz.csv",
        # "long_result_dev0_rodinia_cuda_06022024_163758_mode_0_10_x1_50ms_2000ms_400MHz_2000MHz_400MHz.csv",
        # "long_result_dev0_tango_cuda_05192024_202343_mode_0_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv",
        # "long_result_dev1_tango_cuda_05192024_202341_mode_2_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv",
        # "long_result_dev0_tango_cuda_05202024_040556_mode_3_0_x100_100ms_2000ms_400MHz_2000MHz_400MHz.csv",
    ]
    long_df = None
    # long_df's columns : Iteration,Kernel,Timestamp,Freq,FreqMode,BinPolicy,Power
    for file in files:
        if long_df is None:
            long_df = pd.read_csv(file)
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
    long_df = long_df.fillna("None")

    print(long_df)

    col_names = ["Iteration", "Benchmark", "Kernel", "FreqMode", "Platform", "Device"]
    g_max_df = long_df.groupby(col_names).max().reset_index()
    print(g_max_df)
    g_mean_df = long_df.groupby(col_names).mean().reset_index()

    print(g_mean_df)
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
    target_list = ["Power", "Performance", "EDP", "Freq"]
    g_norm_df = (
        g_mean_df.groupby(["Benchmark", "Kernel", "Iteration", "Platform", "Device"])
        .apply(normalized, targets=target_list)
        .reset_index(drop=True)
    )
    print(g_norm_df)

    # for tgt in target_list:
    #     draw_bar_multigraph(g_norm_df, out_dir / f"bar_graph_{tgt}", tgt)

    # modes = ["base", "random", "HertzPatch"]
    # for tgt in modes:
    #     out_file = out_dir / f"line_graph_power_{tgt}"
    #     new_df = long_df.loc[long_df["FreqMode"] == tgt]
    #     draw_multigraph(new_df, out_file)
    out_file = out_dir / "line_graph_power_All"
    draw_line_multigraph(long_df, out_file, "Timestamp", "Power", "Device", "Benchmark")

    # modes = ["base", "random", "HertzPatch"]
    # for tgt in modes:
    #     out_file = out_dir / f"line_graph_freq_{tgt}"
    #     new_df = long_df.loc[long_df["FreqMode"] == tgt]
    #     draw_multigraph(new_df, out_file, "Freq")
    # out_file = out_dir / "line_graph_freq_All"
    # draw_line_multigraph(long_df, out_file, "Timestamp","Freq","FreqMode","Benchmark")
