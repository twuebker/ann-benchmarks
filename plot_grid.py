import matplotlib as mpl
mpl.use("pgf")
import argparse
import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import compute_metrics, create_linestyles, create_pointset
from ann_benchmarks.results import get_unique_algorithms, load_all_results

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "axes.grid": False,
})

DATASETS = ["gist-960-euclidean", "glove-100-angular", "sift-128-euclidean", "deep-image-96-angular"]
DATASET_LABELS = ["GIST", "GloVe", "SIFT", "Deep"]

NON_ZERO_START_METRICS = {"k-nn", "epsilon", "largeepsilon", "rel", "p50", "p95", "p99", "p999"}


def plot_single(ax, all_data, xn, yn, x_scale, y_scale, linestyles, title, force_zero):
    xm, ym = metrics[xn], metrics[yn]

    def mean_y(algo):
        xs, ys, *_ = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    all_xs, all_ys = [], []
    min_x, max_x = 1, 0
    handles, labels = [], []

    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        all_xs.extend(xs)
        all_ys.extend(ys)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = ax.plot(xs, ys, "-", label=algo, color=color, ms=5, mew=2, lw=1.5, marker=marker)
        handles.append(handle)
        labels.append(algo)

    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])
    ax.set_title(title)

    if x_scale[0] == "a":
        alpha = float(x_scale[1:])
        fun = lambda x: 1 - (1 - x) ** (1 / alpha)
        inv_fun = lambda x: 1 - (1 - x) ** alpha
        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ax.set_xticks([inv_fun(x) for x in np.arange(0, 1.2, 0.2)])
        else:
            from matplotlib import ticker
            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            ax.set_xticks([0, 1/2, 1-1e-1, 1-1e-2, 1-1e-3, 1-1e-4, 1])
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    ax.grid(visible=True, which="major", color="0.65", linestyle="-")

    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        if force_zero or (xn not in NON_ZERO_START_METRICS and x_scale == "linear"):
            x0 = 0
        ax.set_xlim(x0, x1)
    elif x_scale == "logit":
        ax.set_xlim(min_x, max_x)
    elif x_scale == "linear":
        if force_zero or xn not in NON_ZERO_START_METRICS:
            ax.set_xlim(left=0)
        elif all_xs:
            start = max(0, np.floor(min(all_xs) * 10) / 10 - 0.1)
            ax.set_xlim(left=start)

    if "lim" in ym:
        y0, y1 = ym["lim"]
        if force_zero or (yn not in NON_ZERO_START_METRICS and y_scale == "linear"):
            y0 = 0
        ax.set_ylim(y0, y1)
    elif y_scale in ("linear", "symlog"):
        if force_zero or yn not in NON_ZERO_START_METRICS:
            ax.set_ylim(bottom=0)
    elif y_scale == "log" and all_ys:
        ax.set_ylim(bottom=min(y for y in all_ys if y > 0) * 0.5)

    ax.spines["bottom"]._adjust_location()

    return handles, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", default=10)
    parser.add_argument("--definitions", default="algos.yaml")
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output")
    parser.add_argument("-x", "--x-axis", choices=metrics.keys(), default="k-nn")
    parser.add_argument("-y", "--y-axis", choices=metrics.keys(), default="qps")
    parser.add_argument("-X", "--x-scale", default="linear")
    parser.add_argument("-Y", "--y-scale", choices=["linear", "log", "symlog", "logit"], default="linear")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--force-zero", action="store_true")
    parser.add_argument("--format", choices=["pgf", "png"], default="pgf")
    args = parser.parse_args()

    if not args.output:
        ext = ".pgf" if args.format == "pgf" else ".png"
        args.output = "results/grid%s" % ext

    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    linestyles = create_linestyles(sorted(unique_algorithms))

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2))
    axes = axes.flatten()

    legend_handles, legend_labels = None, None

    for i, (dataset_name, label) in enumerate(zip(DATASETS, DATASET_LABELS)):
        dataset, _ = get_dataset(dataset_name)
        results = load_all_results(dataset_name, count, args.batch)
        runs = compute_metrics(np.array(dataset["distances"]), results, args.x_axis, args.y_axis, args.recompute)
        if not runs:
            raise Exception("Nothing to plot for %s" % dataset_name)
        handles, labels = plot_single(
            axes[i], runs, args.x_axis, args.y_axis,
            args.x_scale, args.y_scale, linestyles, label, args.force_zero
        )
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels

    fig.legend(legend_handles, legend_labels, loc="upper center",
               ncol=min(len(legend_labels), 6), frameon=True, fancybox=False,
               shadow=False, framealpha=0.9, prop={"size": 8},
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if args.output.endswith(".pgf"):
        fig.savefig(args.output, bbox_inches="tight")
    else:
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close()