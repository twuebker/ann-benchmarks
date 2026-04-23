"""
Cardinality Estimation Analysis
================================
Evaluates CSVs with columns [Label, Prediction] where both values have been
offset by +1 (i.e., a stored value of 1.0 means true cardinality 0).

Usage
-----
Edit the RUNS list at the bottom of this file, then run:

    python analyze_cardinality.py

Each entry in RUNS is:
    ("Estimator name", "Dataset name", "path/to/file.csv")

The script prints:
  - Per-run confusion matrix and classification report (console)
  - A single LaTeX table combining all runs (stdout, copy into your thesis)
"""

import sys
import math
import csv
from sklearn.metrics import confusion_matrix, classification_report as skl_report


# ── helpers ──────────────────────────────────────────────────────────────────

def load(path: str) -> list[tuple[int, int]]:
    """Return list of (true_label, predicted_label) integer pairs."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            true = float(line["Label"]) - 1
            pred = float(line["Prediction"]) - 1
            rows.append((int(round(true)), max(math.floor(pred), 0)))
    return rows


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "N/A"


# ── per-run analysis ──────────────────────────────────────────────────────────

def compute_metrics(path: str) -> dict:
    """Return a dict of metrics for one CSV file."""
    data = load(path)
    n    = len(data)
    y_true = [0 if t == 0 else 1 for t, _ in data]
    y_pred = [0 if p == 0 else 1 for _, p in data]

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    r = skl_report(y_true, y_pred, output_dict=True,
                   target_names=["zero", "non-zero"])
    return dict(
        n=n,
        TN=int(TN), FP=int(FP), FN=int(FN), TP=int(TP),
        # zero class
        prec_0  = r["zero"]["precision"],
        rec_0   = r["zero"]["recall"],
        f1_0    = r["zero"]["f1-score"],
        # non-zero class
        prec_1  = r["non-zero"]["precision"],
        rec_1   = r["non-zero"]["recall"],
        f1_1    = r["non-zero"]["f1-score"],
        accuracy= r["accuracy"],
    )


def print_run(estimator: str, dataset: str, m: dict) -> None:
    section(f"{estimator}  ·  {dataset}")
    col_w = 14
    print(f"\n  {'':20} {'Pred = 0':>{col_w}} {'Pred ≥ 1':>{col_w}}")
    print(f"  {'True = 0':20} {m['TN']:>{col_w}} {m['FP']:>{col_w}}")
    print(f"  {'True ≥ 1':20} {m['FN']:>{col_w}} {m['TP']:>{col_w}}")
    print()
    print(f"  {'':25} {'zero':>8} {'non-zero':>10}")
    print(f"  {'Precision':25} {m['prec_0']:>8.4f} {m['prec_1']:>10.4f}")
    print(f"  {'Recall':25} {m['rec_0']:>8.4f} {m['rec_1']:>10.4f}")
    print(f"  {'F1':25} {m['f1_0']:>8.4f} {m['f1_1']:>10.4f}")
    print(f"\n  Accuracy: {m['accuracy']:.4f}  ({pct(m['TN']+m['TP'], m['n'])})")


# ── latex output ──────────────────────────────────────────────────────────────

def fmt(v: float) -> str:
    """Format a metric value for LaTeX (4 decimal places)."""
    return f"{v:.4f}"


def build_latex(runs: list[tuple[str, str, dict]], datasets: list[str]) -> str:
    """
    Build a LaTeX table.

    Rows    = estimators
    Columns = dataset groups × metrics
    Metrics = Prec / Rec / F1  for each class (zero and non-zero)
    """
    n_ds      = len(datasets)
    # 6 metric columns per dataset: P0, R0, F1_0, P1, R1, F1_1
    col_spec  = "l" + " ccc ccc" * n_ds

    # --- header rows ---------------------------------------------------------
    # Row 1: dataset multicolumns
    ds_headers = " & ".join(
        f"\\multicolumn{{6}}{{c}}{{\\textbf{{{ds}}}}}" for ds in datasets
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i*6}-{7 + i*6}}}" for i in range(n_ds)
    )

    # Row 2: class sub-headers (zero / non-zero, each spanning 3 cols)
    class_headers = " & ".join(
        "\\multicolumn{3}{c}{\\textit{zero}} & \\multicolumn{3}{c}{\\textit{non-zero}}"
        for _ in datasets
    )
    cmidrules2 = ""
    for i in range(n_ds):
        base = 2 + i * 6
        cmidrules2 += f" \\cmidrule(lr){{{base}-{base+2}}} \\cmidrule(lr){{{base+3}-{base+5}}}"

    # Row 3: metric names
    metric_header = " & ".join(
        "\\textbf{P} & \\textbf{R} & \\textbf{F1} & "
        "\\textbf{P} & \\textbf{R} & \\textbf{F1}"
        for _ in datasets
    )

    # --- data rows -----------------------------------------------------------
    # Group runs by estimator, preserve order
    estimators = list(dict.fromkeys(e for e, _, _ in runs))
    data_rows  = []
    for est in estimators:
        cells = [f"\\textsc{{{est}}}"]
        for ds in datasets:
            m = next((m for e, d, m in runs if e == est and d == ds), None)
            if m:
                cells += [fmt(m["prec_0"]), fmt(m["rec_0"]),  fmt(m["f1_0"]),
                          fmt(m["prec_1"]), fmt(m["rec_1"]),  fmt(m["f1_1"])]
            else:
                cells += ["--"] * 6
        data_rows.append(" & ".join(cells) + " \\\\")

    # --- assemble ------------------------------------------------------------
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Zero vs.\ non-zero cardinality classification: precision (P), recall (R), and F1 per estimator and dataset.}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f" & {ds_headers} \\\\",
        cmidrules,
        f" & {class_headers} \\\\",
        cmidrules2,
        f"\\textbf{{Estimator}} & {metric_header} \\\\",
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:zero_classification}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────

# ╔══════════════════════════════════════════════════════════════╗
# ║  EDIT THIS LIST — one entry per (estimator, dataset, file)  ║
# ╚══════════════════════════════════════════════════════════════╝
RUNS = [
    ("Sampling", "GIST", "./predictions/predictions_Sampling_gist_r0.5.csv"),
    ("Sampling", "SIFT", "./predictions/predictions_Sampling_sift_r10000.csv"),
    ("SelNet", "GIST", "./predictions/predictions_SelNet_gist_r0.5.csv"),
    ("SelNet", "SIFT", "./predictions/predictions_SelNet_sift_r10000.csv"),
]

if __name__ == "__main__":
    results = []
    for estimator, dataset, path in RUNS:
        try:
            m = compute_metrics(path)
        except FileNotFoundError:
            print(f"  [skip] file not found: {path}", file=sys.stderr)
            continue
        print_run(estimator, dataset, m)
        results.append((estimator, dataset, m))

    if not results:
        print("No results to tabulate.", file=sys.stderr)
        sys.exit(1)

    datasets   = list(dict.fromkeys(d for _, d, _ in results))
    print("\n\n" + "═" * 60)
    print("  LaTeX table")
    print("═" * 60 + "\n")
    print(build_latex(results, datasets))
    print()