"""Turn sweep summaries into report figures (PDF + PNG) and LaTeX tables.

    python scripts/plot_results.py state_tracking      # reads results/state_tracking/summary.csv

Outputs go to ``report/figures/`` and ``report/tables/``. With several seeds in the CSV, lines show
the mean and a shaded min-max band, and table cells read ``mean ± std``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR, TAB_DIR = ROOT / "report" / "figures", ROOT / "report" / "tables"

# Reference palette, categorical slots 1-3 (validated all-pairs for small multiples).
# Hue = architecture family; line style = eigenvalue range (secondary, colour-independent encoding).
FAMILY_COLOR = {"ssm": "#2a78d6", "deltanet": "#eb6834", "attention": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

MODELS = {  # label in CSV -> (display name, family)
    "attention": ("Transformer", "attention"),
    "ssm": (r"SSM, $a\in(0,1)$", "ssm"),
    "ssm_neg": (r"SSM, $a\in(-1,1)$", "ssm"),
    "deltanet": (r"Gated DeltaNet, $\beta\in(0,1)$", "deltanet"),
    "deltanet_neg": (r"Gated DeltaNet, $\beta\in(0,2)$", "deltanet"),
    # extensions
    "ssm_rot": (r"Rotational SSM (Mamba-3 style)", "ssm"),
    "deltaproduct2": (r"DeltaProduct$_2$, $\beta\in(0,1)$", "deltanet"),
    "deltaproduct2_neg": (r"DeltaProduct$_2$, $\beta\in(0,2)$", "deltanet"),
}
EXTENSIONS = ("ssm_rot", "deltaproduct2", "deltaproduct2_neg")
# Figure 1 -- baselines: dashed/square = positive eigenvalues only, solid/circle = extended range.
FIG_BASELINE = {"attention": ("-", "o"), "ssm": ("--", "s"), "ssm_neg": ("-", "o"),
                "deltanet": ("--", "s"), "deltanet_neg": ("-", "o")}
# Figure 2 -- extensions vs. their predecessors: dashed/circle = negative eigenvalues (previous best),
# solid/triangle = new transition (rotation / product of reflections).
FIG_EXTENSION = {"ssm_neg": ("--", "o"), "ssm_rot": ("-", "^"),
                 "deltanet_neg": ("--", "o"), "deltaproduct2_neg": ("-", "^")}
GROUPS = {"Z2": (r"$\mathbb{Z}_2$ (parity)", 2), "Z3": (r"$\mathbb{Z}_3$ (count mod 3)", 3), "S3": (r"$S_3$ (permutations)", 6)}
LENGTHS = [("main", 64), ("len256", 256), ("len512", 512)]


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def read_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def aggregate(rows, key_fields, value_field):
    """{key: [values over seeds]}"""
    out = defaultdict(list)
    for r in rows:
        if r.get(value_field) not in (None, ""):
            out[tuple(r[k] for k in key_fields)].append(float(r[value_field]))
    return out


def fmt(values, bold=False):
    if not values:
        return "--"
    s = f"{mean(values):.3f}" if len(values) == 1 else f"{mean(values):.3f}$\\pm${pstdev(values):.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def plot_panels(acc, styles: dict, name: str, n_seeds: int):
    """Accuracy vs evaluation length, one panel per group. ``styles``: model -> (linestyle, marker)."""
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    xs = [length for _, length in LENGTHS]
    for ax, (group, (title, order)) in zip(axes, GROUPS.items()):
        style_axes(ax)
        ax.axhline(1 / order, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
        ax.text(46, 1 / order - 0.015, "chance", color=MUTED, fontsize=7, ha="left", va="top")
        ax.axvspan(40, 90, color=GRID, alpha=0.5, linewidth=0)
        for model, (ls, marker) in styles.items():
            label, family = MODELS[model]
            series = [acc[n].get((model, group), []) for n, _ in LENGTHS]
            if not all(series):
                continue
            ax.plot(xs, [mean(v) for v in series], color=FAMILY_COLOR[family], linestyle=ls, linewidth=2,
                    marker=marker, markersize=6.5, markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
            if n_seeds > 1:
                ax.fill_between(xs, [min(v) for v in series], [max(v) for v in series],
                                color=FAMILY_COLOR[family], alpha=0.12, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_ylim(0, 1.04)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("test sequence length", fontsize=8.5, color=INK_2)
    axes[0].set_ylabel("token accuracy", fontsize=8.5, color=INK_2)
    axes[0].text(64, 0.02, "train length", color=MUTED, fontsize=7, ha="center", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def state_tracking(csv_path: Path):
    rows = read_rows(csv_path)
    acc = {name: aggregate(rows, ("model", "group"), f"acc_{name}") for name, _ in LENGTHS}
    n_seeds = len({r["seed"] for r in rows})

    plot_panels(acc, FIG_BASELINE, "state_tracking", n_seeds)
    if any(key[0] in EXTENSIONS for key in acc["main"]):
        plot_panels(acc, FIG_EXTENSION, "state_tracking_ext", n_seeds)

    # ---- table ------------------------------------------------------------------------------
    cols = [(g, n) for g in GROUPS for n, _ in LENGTHS]
    best = {c: max((mean(v) for (m, g), v in acc[c[1]].items() if g == c[0]), default=None) for c in cols}
    lines = [
        r"\begin{tabular}{l" + "ccc" * len(GROUPS) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{GROUPS[g][0]}}}" for g in GROUPS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(GROUPS))),
        "Model & " + " & ".join(str(length) for _ in GROUPS for _, length in LENGTHS) + r" \\",
        r"\midrule",
    ]
    present = {m for (m, _) in acc["main"]}
    for model, (name, _) in MODELS.items():
        if model not in present:
            continue
        if model == EXTENSIONS[0]:
            lines.append(r"\midrule")
        cells = []
        for g, n in cols:
            v = acc[n].get((model, g), [])
            cells.append(fmt(v, bold=bool(v) and best[(g, n)] is not None and mean(v) >= best[(g, n)] - 0.005))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "state_tracking.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {FIG_DIR / 'state_tracking.pdf'} and {TAB_DIR / 'state_tracking.tex'} ({n_seeds} seed(s))")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", choices=["state_tracking"])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    {"state_tracking": state_tracking}[args.sweep](args.results / args.sweep / "summary.csv")


if __name__ == "__main__":
    main()
