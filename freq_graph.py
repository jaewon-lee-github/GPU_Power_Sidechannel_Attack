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


def draw_multigraph(input, output_file):
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
    ##  print(fm.findfont("Calibri")) 
    ## print(fm.findSystemFonts(fontpaths=None, fontext="ttf"))

    sns.set_theme(
        "paper", "whitegrid", "Blues", font_scale=1, rc={"font.family": "Calibri"}
    )
    plt.rcParams["font.family"] = "Calibri"
    print(plt.rcParams.keys())

    # sns.set_context('paper', font_scale=1.0)
    # sns.set_style('whitegrid')
    cp = sns.color_palette("Paired", 2)
    # cp = sns.color_palette("crest")
    # cp = sns.cubehelix_palette(n_colors=2, start=.5, rot=-.75)
    # cp = sns.cubehelix_palette(start=.5, rot=-.5)
    # cp = sns.color_palette("Blues")
    # cp = sns.color_palette("viridis")
    print("How many colors in palette?", len(cp))

    sns.axes_style("whitegrid")
    # baseline_color = cp[0]
    # Hertzpatch_color= cp[1]

    # input data structure
    # "Kernel,Timestamp,Freq,Power,Freq_mode"
    # input.to_csv(result_dir / "temp.csv", index=False, mode="w")
    # input['Kernel'] = input['Kernel'].map(lambda x: x.split('_')[-1])
    num_ker = len(input["Kernel"].unique())
    input["Kernel"] = input["Kernel"].map(lambda x: x.replace("_", "\n"))

    g = sns.relplot(
        data=input,
        kind="line",
        x="Timestamp",
        y="Power",
        hue="Freq_mode",
        # style="Freq_mode",
        palette=cp,
        col="Kernel",
        col_wrap=math.ceil(num_ker / 2),
        height=3,
        aspect=0.75,
        linewidth=2,
        facet_kws={"xlim": (0, 100)},
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
    )
    # g = sns.FacetGrid(df, col="Kernel", col_wrap=4, height=2, hue="Freq_mode", palette=cp)
    # g.map(sns.lineplot, x="Timestamp", y="Power", linewidth =1.0, errorbar=None)
    # bbox_to_anchor reference: https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib/39806180#39806180
    g.set_titles(
        col_template="{col_name}", row_template="{row_name}", size=12, fontweight="bold"
    )
    g.set_axis_labels("Time(ms)", "Power (mW)")
    g._legend.set_title("Freq_mode")
    g._legend.texts[0].set_text("Base")
    g._legend.texts[1].set_text("HertzPatch")
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        title=None,
        frameon=True,
        fontsize=12,
    )
    sns.despine(
        top=False, right=False, left=False, bottom=False, offset=None, trim=False )
    # g.set_xticklabels(rotation=45)

    # Set the main title
    # g.fig.suptitle("Scatterplot of Total Bill vs. Tip", y=1.03)
    # handles, labels = axes[-1][0].get_legend_handles_labels()
    # g.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     bbox_to_anchor=(-1.5, 1.05),
    #     ncol=2,
    #     fontsize=12,
    # )

    g.savefig(str(output_file) + ".pdf", format="pdf", bbox_inches="tight", dpi=600)

    return g

    # Set the font size and style for x-axis label and tick labels
    x_label = x_axis.get_label()
    x_label.set_fontsize(9)  # Adjust the font size as needed
    x_ticklabels = x_axis.get_ticklabels()
    for label in x_ticklabels:
        label.set_fontsize(10)  # Adjust the font size for tick labels as needed

    # Set the font size and style for y-axis label and tick labels
    y_label = y_axis.get_label()
    y_label.set_fontsize(12)  # Adjust the font size as needed
    y_ticklabels = y_axis.get_ticklabels()
    for label in y_ticklabels:
        label.set_fontsize(10)

    # Set the font properties for the legend
    for text in legend.texts:
        print(text.get_text())
        text.set_font_properties(font_prop)
    ax_column.set_xlabel("", fontsize=9)
    ax_column.set_title(workload, fontsize=13)

    ax_column.set_ylabel("", fontsize=9)
    ax_column.get_legend().remove()

    # Graph

    return g
    for row, ax_row in enumerate(axes):
        for column, ax_column in enumerate(ax_row):
            idx = row * 2 + column
            workload = workloads[idx]
            sns.lineplot(
                data=org_df[workload],
                ax=ax_column,
                label="Baseline",
                color=baseline_color,
                linewidth=1.0,
            )
            sns.lineplot(
                data=dvfs_df[workload],
                ax=ax_column,
                label="HertzPatch",
                color=Hertzpatch_color,
                linewidth=1.0,
            )

            # thickness=1.0
            # for p in ['left', 'right', 'top', 'bottom']:
            #   plt.gca().spines[p].set_linewidth(thickness)
            #   plt.gca().spines[p].set_color('black')
            # # ax_column.set_yticks("")
            # ax_column.
            # ax_column.set_yticks([-1,200,400,1000],fontsize=10)
            # ax_column.xticks(fontsize=9)
            # ax_column.yticks(fontsize=9)
            ax_column.set_xlabel("", fontsize=9)
            # ax_column.set_ylabel('Power (W)',fontsize=9)
            ax_column.set_title(workload, fontsize=13)

            ax_column.set_ylabel("", fontsize=9)
            ax_column.get_legend().remove()
            # ax_column.

    handles, labels = axes[-1][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(-1.5, 1.05),
        ncol=2,
        fontsize=12,
    )
    fig.text(-1.5, 0.02, "Time (* 10K Cycle)", ha="center", fontsize=15)
    fig.text(-1, 0.5, "Power (W)", va="center", rotation="vertical", fontsize=15)
    fig.tight_layout()

    output_file = "Powertrace_Result"
    output_file = output_dir / output_file
    fig.savefig(output_file + ".pdf", format="pdf", bbox_inches="tight", dpi=600)


def collect_data(benchmark, freq_mode, interval):
    acc_df = None
    cur_dir = os.getcwd()
    os.chdir(benchmark.base_dir)
    for file in glob.iglob(
        f"./**/output_{freq_mode}_*_{interval}ms.csv", recursive=True
    ):
        print(f"current file: {file}")
        df = pd.read_csv(file)
        if acc_df is None:
            acc_df = df
        else:
            acc_df = pd.concat([acc_df, df], ignore_index=True, axis=0)
        # os.remove(file)
    if acc_df is None:
        print("ERR: No result file")
        exit()

    new_column = acc_df.columns.tolist() + ["Freq_mode"]
    acc_df = acc_df.reindex(columns=new_column, fill_value=freq_mode)
    os.chdir(cur_dir)
    return acc_df


def collect_data_freq_mode(benchmark, freq_mode_list, interval):
    res_df = None
    for i in freq_mode_list:
        df = collect_data(benchmark, i, interval)
        if res_df is None:
            res_df = df
        else:
            res_df = pd.concat([res_df, df], ignore_index=True, axis=0)

    threshold_sec = 4  # Change this to your desired threshold
    threshold = threshold_sec * 1000 / interval
    value_counts = res_df["Kernel"].value_counts()
    mask = res_df["Kernel"].map(value_counts) >= threshold
    filt_acc_df = res_df[mask]
    # filt_acc_df= acc_df.groupby("Kernel").filter(lambda group: len(group) >= threshold)
    # Step 3: Use the boolean mask to filter the DataFrame

    filt_wide_df = filt_acc_df.pivot(
        index=["Kernel", "Freq_mode"], columns="Timestamp", values="Power"
    )
    filt_wide_df.reset_index(inplace=True)
    # filt_acc_df.to_csv(f"{result_dir}/acc_df_{freq_mode}_{interval}ms.csv", index=False, mode="w")
    # filt_wide_df.to_csv(f"{result_dir}/wide_df_{freq_mode}_{interval}ms.csv", index=False, mode="w")
    return filt_acc_df, filt_wide_df


if __name__ == "__main__":
    benchmark = Benchmark("tango_cuda")

    # freq_mode 0 = Natural, 1 = Random DVFS
    # load orginal dataframe
    interval = 100
    freq_mode = [0, 1]
    long_df, wide_df = collect_data_freq_mode(benchmark, freq_mode, interval)
    # df = pd.read_csv(f"{result_dir}/acc_df_{freq_mode}_{interval}ms.csv")

    data_dir = result_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    out_dir = result_dir / "../figures"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir / "Powertrace_Result"

    draw_multigraph(long_df, out_file)

    Perf_File = "Perf.csv"

    Perf_df = pd.read_csv(os.path.join(data_dir, Perf_File))
    Perf_df["Workload"] = Perf_df["Workload"].str.lower()

    sns.set_context("paper", font_scale=2.0)
    sns.set_style("whitegrid")

    color = sns.color_palette("Paired")[1]
    # n_columns= len(Fig9_df['Mechanism'].unique())
    # print(n_columns)
    # n_workloads = len(Fig9_df['Workload'].unique())

    # Legend_order = []

    # Fig9_df = Fig9_df[ Fig9_df['Mechanism']!='Baseline']

    # fig, ax = plt.subplots(figsize=(5, 6))
    # paper_textwidth = 345

    # ha = ['right', 'center', 'left']

    fig, ax = plt.subplots(figsize=(8, 2.5))
    p = sns.barplot(
        data=Perf_df,
        x=Perf_df["Workload"],
        y=Perf_df["Perf"],
        edgecolor="#192133",
        linewidth=2.0,
        color=color,
    )

    # plt.title('Params : {}'.format(name), y = 1.10,fontsize=14)

    handles, labels = ax.get_legend_handles_labels()

    # ax.legend(handles, labels,
    #         frameon=False,
    #         fancybox=None,
    #         columnspacing=0.6,
    #         facecolor=None, edgecolor=None,
    #         bbox_to_anchor=(0.5, 1.25), loc=10, ncol=n_columns,fontsize=14)

    y_limit = 0.0
    ax.set_ylim(-1, y_limit)

    # ax.set_y
    ax.set_yticks([-1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Norm. Ratio")
    ax.set_xlabel("")
    plt.xticks(rotation=39, ha="right")
    thickness = 1.0
    for p in ["left", "right", "top", "bottom"]:
        plt.gca().spines[p].set_linewidth(thickness)
        plt.gca().spines[p].set_color("black")

    plt.grid(True, linestyle="--", dashes=(5, 6), linewidth=1.5, axis="y")
    # ax.yaxis.grid(True) # Hide the horizontal gridlines

    # fig.tight_layout()
    out_file = "Perf_Improvement"
    out_file = os.path.join(Output_dir, out_file)
    # fig.savefig(output_file+".png", format='png',bbox_inches='tight',dpi=599)
    fig.savefig(out_file + ".eps", format="eps", bbox_inches="tight", dpi=599)

    Workload_num = len(workloads)
    print(Workload_num)
