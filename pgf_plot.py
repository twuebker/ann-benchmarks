import matplotlib as mpl

# Use PGF backend for LaTeX-style plots
mpl.use("pgf")
import argparse

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (compute_metrics, create_linestyles,
                                           create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results

# Configure matplotlib for LaTeX-style output
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
    "axes.grid": False,  # Disable grid by default
})

# Metrics that should NOT start at zero (ratios, percentages, latencies)
NON_ZERO_START_METRICS = {
    "k-nn",  # Recall - meaningful range is usually 0.6-1.0
    "epsilon",  # Epsilon recall
    "largeepsilon",  # Large epsilon recall
    "rel",  # Relative error - ratio
    "p50", "p95", "p99", "p999",  # Latency percentiles - never zero in practice
}


def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, batch, force_zero):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(6, 4))  # Reduced from (12, 9) for better thesis integration

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    # Collect all data points to determine appropriate ranges
    all_xs, all_ys = [], []

    # Find range for logit x-scale and collect data
    min_x, max_x = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        all_xs.extend(xs)
        all_ys.extend(ys)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = plt.plot(
            xs, ys, "-", label=algo, color=color, ms=5, mew=2, lw=1.5, marker=marker
        )  # Reduced line/marker sizes for cleaner look
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                axs, ays, "-", label=algo, color=faded, ms=3, mew=1, lw=1, marker=marker
            )
        labels.append(algo)

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])

    # Custom scales of the type --x-scale a3
    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def fun(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_fun(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(get_plot_label(xm, ym))

    # Legend inside plot area for better thesis integration
    ax.legend(handles, labels, loc="best", frameon=True, fancybox=False,
              shadow=False, framealpha=0.9, prop={"size": 7})

    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Smart axis limits - context-aware approach
    # X-axis limits
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        # Only force zero if explicitly requested OR if it's a count/quantity metric
        if force_zero or (xn not in NON_ZERO_START_METRICS and x_scale == "linear"):
            x0 = 0
        plt.xlim(x0, x1)
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    elif x_scale == "linear":
        # For linear scale without limits, be smart about starting point
        if force_zero or xn not in NON_ZERO_START_METRICS:
            plt.xlim(left=0)
        else:
            # For recall-like metrics, start slightly below minimum data
            if all_xs:
                data_min = min(all_xs)
                # Start at a nice round number below data_min
                start = max(0, np.floor(data_min * 10) / 10 - 0.1)
                plt.xlim(left=start)

    # Y-axis limits
    if "lim" in ym:
        y0, y1 = ym["lim"]
        # Only force zero if explicitly requested OR if it's a count/quantity metric
        if force_zero or (yn not in NON_ZERO_START_METRICS and y_scale == "linear"):
            y0 = 0
        plt.ylim(y0, y1)
    elif y_scale == "linear" or y_scale == "symlog":
        # For linear scale without limits, be smart about starting point
        if force_zero or yn not in NON_ZERO_START_METRICS:
            plt.ylim(bottom=0)
        else:
            # For latency/recall metrics, use matplotlib's auto-scaling
            # which will choose a reasonable minimum
            pass
    elif y_scale == "log":
        # Log scale should never include zero
        if all_ys:
            y_min = min([y for y in all_ys if y > 0])
            plt.ylim(bottom=y_min * 0.5)  # Start slightly below minimum

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    # Save as PGF for LaTeX, or PNG if filename ends with .png
    if fn_out.endswith('.pgf'):
        plt.savefig(fn_out, bbox_inches="tight")
    else:
        plt.savefig(fn_out, bbox_inches="tight", dpi=300)  # Higher DPI for thesis quality
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular")
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-x", "--x-axis", help="Which metric to use on the X-axis", choices=metrics.keys(), default="k-nn"
    )
    parser.add_argument(
        "-y", "--y-axis", help="Which metric to use on the Y-axis", choices=metrics.keys(), default="qps"
    )
    parser.add_argument(
        "-X", "--x-scale", help="Scale to use when drawing the X-axis. Typically linear, logit or a2", default="linear",
    )
    parser.add_argument(
        "-Y",
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )
    parser.add_argument("--batch", help="Plot runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    parser.add_argument(
        "--grid", help="Enable grid lines", action="store_true"
    )
    parser.add_argument(
        "--format", help="Output format (pgf for LaTeX, png for raster)",
        choices=["pgf", "png"], default="png"
    )
    parser.add_argument(
        "--force-zero", help="Force all axes to start at zero (overrides smart defaults)",
        action="store_true"
    )
    args = parser.parse_args()

    # Enable grid if requested
    if args.grid:
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
        plt.rcParams["grid.linestyle"] = ":"

    if not args.output:
        ext = ".pgf" if args.format == "pgf" else ".png"
        args.output = "results/%s%s" % (args.dataset + ("-batch" if args.batch else ""), ext)
        print("writing output to %s" % args.output)

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, args.batch)
    linestyles = create_linestyles(sorted(unique_algorithms))
    runs = compute_metrics(np.array(dataset["distances"]), results, args.x_axis, args.y_axis, args.recompute)
    if not runs:
        raise Exception("Nothing to plot")

    create_plot(
        runs, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles, args.batch,
        args.force_zero
    )