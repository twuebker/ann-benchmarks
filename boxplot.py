"""
plot_cardinality.py
--------------------
Generates Leis-et-al.-style boxplots for similarity search cardinality estimation.

Expected file naming convention (edit FILE_PATTERN below if yours differs):
    {dataset}_{model}_r{radius}.csv          e.g.  SIFT_ModelA_r1.csv
    or place files in a folder and set DATA_DIR.

Each CSV must have a header line:
    Label,Prediction
Values have 1 added (1.0 means 0 true count). The script corrects for this.

Output: one PDF per (dataset, model) combination → 4 PDFs total,
        plus one combined figure with all 4 panels side by side.

Metric plotted on y-axis:
    signed log2 ratio = log2(prediction / true)
        0  → perfect estimate
       >0  → overestimate
       <0  → underestimate
    When true == 0 (and prediction == 0) the error is 0.
    When true == 0 but prediction > 0 we use log2(prediction + 1) as a proxy.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embeds fonts properly for submission
})

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit these to match your files
# ─────────────────────────────────────────────

DATA_DIR = "./predictions"          # folder containing your 12 CSV files

# File naming: adjust the regex to match your actual filenames.
# Expected groups: dataset, model, radius
# Example pattern matches:  SIFT_ModelA_r0.1.csv
FILE_PATTERN = r"predictions_(?P<model>[^_]+)_(?P<dataset>sift|gist)_r(?P<radius>.+)\.csv"

# If you'd rather list files explicitly, set EXPLICIT_FILES to a list of dicts:
# EXPLICIT_FILES = [
#     {"path": "SIFT_ModelA_r0.1.csv", "dataset": "SIFT", "model": "ModelA", "radius": "0.1"},
#     ...
# ]
EXPLICIT_FILES = None   # set to None to use auto-discovery via FILE_PATTERN

OUTPUT_DIR = "."        # where to save PDFs

# Whisker positions (percentiles)
WHISKER_LO = 5
WHISKER_HI = 95

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_file(path):
    """Load a CSV, subtract 1 from both columns, floor predictions to int ≥ 0."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "prediction"})
    df["label"] = df["label"] - 1.0          # true cardinality
    df["prediction"] = np.maximum(np.floor(df["prediction"] - 1.0), 0.0)
    return df


def signed_log2_ratio(true, pred):
    """
    Signed log2 ratio: log2(pred / true).
    Handles zeros:
      true == 0, pred == 0  → error = 0  (perfect)
      true == 0, pred  > 0  → error = +log2(pred + 1)  (overestimate proxy)
      true  > 0, pred == 0  → error = -log2(true + 1)  (underestimate proxy)
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    result = np.zeros_like(true)
    both_zero = (true == 0) & (pred == 0)
    over_zero  = (true == 0) & (pred > 0)
    under_zero = (true > 0)  & (pred == 0)
    normal     = (true > 0)  & (pred > 0)
    result[over_zero]  =  np.log2(pred[over_zero]  + 1)
    result[under_zero] = -np.log2(true[under_zero] + 1)
    result[normal]     =  np.log2(pred[normal] / true[normal])
    return result


def discover_files():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    records = []
    for path in csvs:
        fname = os.path.basename(path)
        m = re.search(FILE_PATTERN, fname, re.IGNORECASE)
        if m:
            records.append({
                "path": path,
                "dataset": m.group("dataset").upper(),
                "model": m.group("model"),
                "radius": m.group("radius"),
            })
    if not records:
        raise FileNotFoundError(
            f"No CSV files matched pattern '{FILE_PATTERN}' in '{DATA_DIR}'.\n"
            "Please adjust FILE_PATTERN or set EXPLICIT_FILES."
        )
    return records


def load_all():
    file_list = EXPLICIT_FILES if EXPLICIT_FILES else discover_files()
    data = {}   # (dataset, model) → {radius: [errors]}
    for entry in file_list:
        df = load_file(entry["path"])
        errors = signed_log2_ratio(df["label"].values, df["prediction"].values)
        key = (entry["dataset"], entry["model"])
        data.setdefault(key, {})
        data[key][entry["radius"]] = errors
    return data

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def make_boxplot_stats(errors, whisker_lo=WHISKER_LO, whisker_hi=WHISKER_HI):
    """Return dict of statistics for a custom box."""
    return {
        "median": np.median(errors),
        "q25":    np.percentile(errors, 25),
        "q75":    np.percentile(errors, 75),
        "wlo":    np.percentile(errors, whisker_lo),
        "whi":    np.percentile(errors, whisker_hi),
        "outliers_lo": errors[errors < np.percentile(errors, whisker_lo)],
        "outliers_hi": errors[errors > np.percentile(errors, whisker_hi)],
    }


def draw_panel(ax, radius_data, title, ylim):
    """
    Draw one panel (one dataset × one model) onto ax.
    radius_data: dict { radius_label → np.array of signed log2 errors }
    """
    radii = sorted(radius_data.keys(), key=lambda r: float(r))
    n = len(radii)
    positions = np.arange(n) * 0.8 + 1
    box_width = 0.35
    cap_width = 0.25

    for pos, radius in zip(positions, radii):
        errors = radius_data[radius]
        s = make_boxplot_stats(errors)

        # IQR box
        box = matplotlib.patches.FancyBboxPatch(
            (pos - box_width / 2, s["q25"]),
            box_width,
            s["q75"] - s["q25"],
            boxstyle="square,pad=0",
            linewidth=0.8,
            edgecolor="black",
            facecolor="white",
            zorder=3,
        )
        ax.add_patch(box)
        ax.set_ylim(-ylim, ylim)

        # Median line
        ax.plot([pos - box_width / 2, pos + box_width / 2],
                [s["median"], s["median"]],
                color="black", linewidth=1.2, zorder=4)

        # Whiskers
        ax.plot([pos, pos], [s["q75"], s["whi"]],
                color="black", linewidth=0.8, zorder=3)
        ax.plot([pos, pos], [s["q25"], s["wlo"]],
                color="black", linewidth=0.8, zorder=3)

        # Caps
        for y in [s["wlo"], s["whi"]]:
            ax.plot([pos - cap_width / 2, pos + cap_width / 2],
                    [y, y], color="black", linewidth=0.8, zorder=3)

        # Outlier dots
        for ov in [s["outliers_lo"], s["outliers_hi"]]:
            if len(ov):
                ax.scatter(
                    np.full_like(ov, pos), ov,
                    s=3, color="black", zorder=5, linewidths=0,
                )

    # Reference line at 0 (perfect estimation)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([format_radius(r) for r in radii])
    ax.set_xlim(0.4, n + 0.6)

    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(axis="both", labelsize=8)

    # Custom y-tick labels: show power-of-2 multipliers
    #def y_fmt(val, _):
    #    if val == 0:
    #        return "1×"
    #    elif val > 0:
    #        return f"$2^{{{val:.0f}}}$×"
    #    else:
    #        return f"$1/2^{{{abs(val):.0f}}}$×"

    def y_fmt(val, _):
        if val == 0:
            return "1"
        return f"$2^{{{int(val)}}}$"

    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))  # BIG reduction
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.grid(which="major", axis="both", linewidth=0.4, color="lightgray", zorder=0)
    ax.grid(which="minor", axis="both", linewidth=0.2, color="lightgray", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # new
    ax.set_xlim(positions[0] - 0.3, positions[-1] + 0.3)

def format_radius(r):
    r = float(r)
    if r >= 1000:
        return f"{int(r/1000)}k"
    elif r >= 100:
        return f"{int(r)}"
    elif r >= 1:
        return f"{r:g}"
    else:
        return f"{r:g}"

def make_legend(ax):
    """Add a compact percentile legend."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, FancyBboxPatch
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.2, label='Median'),
        Patch(facecolor='white', edgecolor='black', lw=0.8, label='25th–75th pct.'),
        Line2D([0], [0], color='black', lw=0.8, label=f'{WHISKER_LO}th–{WHISKER_HI}th pct.'),
        Line2D([0], [0], marker='o', color='black', lw=0,
               markersize=3, label='Outliers'),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=6,  # smaller
        loc="upper left",
        frameon=False,
        handlelength=1.2,  # shorter lines
        handletextpad=0.4,  # tighter text spacing
        borderpad=0.2,
        labelspacing=0.2  # tighter vertical spacing
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    data = load_all()

    # Determine all (dataset, model) combos in sorted order
    combos = sorted(data.keys())   # e.g. [("GIST","ModelA"), ("GIST","ModelB"), ...]
    n_panels = len(combos)

    if n_panels == 0:
        print("No data found. Check DATA_DIR and FILE_PATTERN.")
        return

    # ── Combined 4-panel figure ──────────────────────────────────────────────
    ncols = n_panels
    nrows = 1
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(1.8 * ncols, 3.2),
        sharey=True
    )
    fig.subplots_adjust(wspace=0.08)  # very tight spacing
    axes = np.array(axes).flatten()

    # Compute global y-axis limits across ALL panels
    global_max = 0
    for key in data:
        for v in data[key].values():
            lo = abs(np.percentile(v, 1))
            hi = abs(np.percentile(v, 99))
            global_max = max(global_max, lo, hi)

    global_max = max(global_max, 3)
    ylim = global_max * 1.15

    for i, (dataset, model) in enumerate(combos):
        ax = axes[i]
        draw_panel(ax, data[(dataset, model)], title=f"{dataset}: {model}", ylim=ylim)
        if i != 0:
            ax.tick_params(axis='y', which='both', length=0)
        if i == 0:
            ax.set_ylabel("Estimation error  (signed log₂ ratio)", fontsize=8, labelpad=6)
            make_legend(ax)
        else:
            ax.tick_params(axis='y', labelleft=False)  # ← better than set_yticklabels([])
            ax.set_ylabel("")

    # Hide unused panels if n_panels is odd
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel("Search radius", fontsize=9, y=-0.02)

    combined_path = os.path.join(OUTPUT_DIR, "cardinality_boxplots_combined.pdf")
    fig.savefig(combined_path, bbox_inches="tight")
    print(f"Saved: {combined_path}")

    # ── Individual PDFs ──────────────────────────────────────────────────────
    for dataset, model in combos:
        fig2, ax2 = plt.subplots(figsize=(3.3, 3.0))
        draw_panel(ax2, data[(dataset, model)], title=f"{dataset}: {model}", ylim=ylim)
        ax2.set_ylabel("Estimation error  (signed log₂ ratio)", fontsize=8)
        make_legend(ax2)
        fig2.tight_layout()
        fname = f"cardinality_boxplot_{dataset}_{model}.pdf"
        fpath = os.path.join(OUTPUT_DIR, fname)
        fig2.savefig(fpath, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {fpath}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()