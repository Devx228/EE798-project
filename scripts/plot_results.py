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


TRAIN_LEN_COLOR = {"T16": "#86b6ef", "T32": "#2a78d6", "T64": "#104281"}  # ordinal blue ramp, light -> dark


def rotation_length(csv_path: Path):
    """Rotational SSM on Z3: accuracy vs test length for each training length (RQ2b ablation)."""
    rows = read_rows(csv_path)
    n_seeds = len({r["seed"] for r in rows})
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    style_axes(ax)
    ax.axhline(1 / 3, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(15, 1 / 3 - 0.015, "chance", color=MUTED, fontsize=7, ha="left", va="top")
    table = []
    for label, color in TRAIN_LEN_COLOR.items():
        runs = [r for r in rows if r["train_len"] == label]
        if not runs:
            continue
        T = int(label[1:])
        cols = [("acc_main", T)] + [(f"acc_len{L}", L) for L in (64, 256, 512) if L > T]
        xs = [L for _, L in cols]
        vals = [[float(r[c]) for r in runs] for c, _ in cols]
        ax.plot(xs, [mean(v) for v in vals], color=color, linewidth=2, marker="o", markersize=6.5,
                markeredgecolor="white", markeredgewidth=1.2, label=f"trained on length {T}", zorder=3)
        ax.scatter([T], [mean(vals[0])], s=110, facecolor="none", edgecolor=color, linewidth=1.5, zorder=4)
        if n_seeds > 1:
            ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals], color=color, alpha=0.12, linewidth=0)
        table.append((T, {L: v for (_, L), v in zip(cols, vals)}))
    ax.set_xscale("log", base=2)
    ticks = [16, 32, 64, 256, 512]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("test sequence length (ring = training length)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("token accuracy", fontsize=8.5, color=INK_2)
    ax.set_title(r"Rotational SSM on $\mathbb{Z}_3$", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"rotation_length.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    lengths = [16, 32, 64, 256, 512]
    lines = [r"\begin{tabular}{l" + "c" * len(lengths) + "}", r"\toprule",
             "Train length & " + " & ".join(f"test {L}" for L in lengths) + r" \\", r"\midrule"]
    for T, accs in table:
        lines.append(f"{T} & " + " & ".join(fmt(accs[L]) if L in accs else "--" for L in lengths) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "rotation_length.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {FIG_DIR / 'rotation_length.pdf'} and {TAB_DIR / 'rotation_length.tex'} ({n_seeds} seed(s))")


MIXER_LABEL = {"attention": "Attention (SDPA)", "ssm": "Selective SSM (chunked)", "deltanet": "Gated DeltaNet (chunked)"}


def _loglog(ax, title, ylabel, xlabel):
    style_axes(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.6)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)


def benchmark(results_dir: Path):
    """RQ4: training-time and decode-time cost on one GPU (CSV files from scripts/benchmark.py)."""
    layer_csv = next(results_dir.glob("layer_*.csv"))
    decode_csv = next(results_dir.glob("decode_*.csv"))
    scan_csv = next(results_dir.glob("scan_*.csv"))
    gpu = layer_csv.stem.split("_", 1)[1].replace("_", " ")
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
    marker = dict(marker="o", markersize=5.5, markeredgecolor="white", markeredgewidth=1.1, linewidth=2)

    # -- training-time view: latency and memory of one layer vs sequence length --------------------
    rows = read_rows(layer_csv)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    panels = [("fwd_ms", "Forward latency", "ms"), ("fwd_bwd_ms", "Forward + backward latency", "ms"),
              ("fwd_bwd_mem_mb", "Peak memory (fwd + bwd)", "MB")]
    for ax, (col, title, unit) in zip(axes, panels):
        _loglog(ax, title, unit, "sequence length")
        for mixer, label in MIXER_LABEL.items():
            pts = [(int(r["seq_len"]), float(r[col])) for r in rows if r["mixer"] == mixer and r[col]]
            ax.plot(*zip(*pts), color=FAMILY_COLOR[mixer], label=label, **marker)
        lengths = sorted({int(r["seq_len"]) for r in rows})
        ax.set_xticks(lengths, [f"{L // 1024}k" if L >= 1024 else str(L) for L in lengths])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"efficiency_layer.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # -- inference-time view: per-token decode latency and memory held per sequence ----------------
    rows = read_rows(decode_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, (col, title, unit) in zip(axes, [("ms_per_token", "Decode latency per token", "ms"),
                                              ("state_kb", "KV cache / recurrent state", "KB")]):
        _loglog(ax, title, unit, "context length (tokens)")
        for mixer, label in MIXER_LABEL.items():
            pts = [(int(r["context"]), float(r[col])) for r in rows if r["mixer"] == mixer]
            ax.plot(*zip(*pts), color=FAMILY_COLOR[mixer], label=label, **marker)
        ctx = sorted({int(r["context"]) for r in rows})
        ax.set_xticks(ctx, [str(c) for c in ctx])
    axes[0].set_ylim(0.5, 10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"efficiency_decode.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # -- scan backends table (lengths >= 4096: the first run's shorter lengths were clock-affected) --
    rows = [r for r in read_rows(scan_csv) if int(r["seq_len"]) >= 4096]
    lengths = sorted({int(r["seq_len"]) for r in rows})
    get = {(r["backend"], int(r["seq_len"])): r for r in rows}
    lines = [r"\begin{tabular}{l" + "c" * len(lengths) + "}", r"\toprule",
             "Backend & " + " & ".join(f"$T={L // 1024}$k" for L in lengths) + r" \\", r"\midrule"]
    for backend, label in [("chunked", "Chunked PyTorch (ms)"), ("triton", "Triton kernel (ms)")]:
        lines.append(f"{label} & " + " & ".join(f"{float(get[(backend, L)]['ms']):.1f}" if (backend, L) in get else "--"
                                                for L in lengths) + r" \\")
    lines.append("Speed-up & " + " & ".join(
        f"{float(get[('chunked', L)]['ms']) / float(get[('triton', L)]['ms']):.2f}$\\times$" for L in lengths) + r" \\")
    for backend, label in [("chunked", "Chunked peak memory (MB)"), ("triton", "Triton peak memory (MB)")]:
        lines.append(f"{label} & " + " & ".join(f"{float(get[(backend, L)]['peak_mem_mb']):.0f}" for L in lengths) + r" \\")
    lines.append("Triton bandwidth (GB/s) & " + " & ".join(
        f"{float(get[('triton', L)]['achieved_GBps']):.1f}" for L in lengths) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "scan_backends.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote efficiency_layer, efficiency_decode figures and scan_backends table ({gpu})")


RECALL_MODELS = {  # label -> (display name, colour, marker); SSM state sizes share the blue family
    "attention_conv": ("Attention + short conv (growing KV cache)", FAMILY_COLOR["attention"], "o"),
    "ssm": ("SSM, N=16 (8k-float state)", "#86b6ef", "s"),
    "ssm_N64": ("SSM, N=64 (33k-float state)", "#104281", "s"),
    "deltanet": ("Gated DeltaNet (16k-float state)", FAMILY_COLOR["deltanet"], "D"),
}


def recall_capacity(csv_path: Path):
    """RQ1: MQAR accuracy vs number of key-value pairs (figure + table with steps-to-90%)."""
    rows = read_rows(csv_path)
    kvs = sorted({int(r["kv"][2:]) for r in rows})
    get = {(r["model"], int(r["kv"][2:])): r for r in rows}
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    style_axes(ax)
    for model, (label, color, marker) in RECALL_MODELS.items():
        pts = [(kv, float(get[(model, kv)]["acc_main"])) for kv in kvs if (model, kv) in get]
        if pts:
            ax.plot(*zip(*pts), color=color, marker=marker, markersize=6.5, markeredgecolor="white",
                    markeredgewidth=1.2, linewidth=2, label=label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(kvs, [str(k) for k in kvs])
    ax.minorticks_off()
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("key-value pairs to remember", fontsize=8.5, color=INK_2)
    ax.set_ylabel("recall accuracy", fontsize=8.5, color=INK_2)
    ax.set_title("Associative recall (MQAR) vs. memory demand", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"recall_capacity.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    lines = [r"\begin{tabular}{lr" + "cc" * len(kvs) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{kv} pairs}}" for kv in kvs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(kvs))),
             "Model & State & " + " & ".join("acc. & steps" for _ in kvs) + r" \\", r"\midrule"]
    names = {"attention_conv": "Attention + conv", "ssm": "SSM, $N{=}16$", "ssm_N64": "SSM, $N{=}64$",
             "deltanet": "Gated DeltaNet"}
    for model, name in names.items():
        state = next((get[(model, kv)]["state_size"] for kv in kvs if (model, kv) in get), "")
        state = f"{int(state):,}" if state else "$O(T)$"
        cells = []
        for kv in kvs:
            r = get.get((model, kv))
            if r is None:
                cells += ["--", "--"]
                continue
            acc = float(r["acc_main"])
            cells += [rf"\textbf{{{acc:.3f}}}" if acc >= 0.99 else f"{acc:.3f}", r["steps_to_90"] or r"$>$20k"]
        lines.append(f"{name} & {state} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "recall_capacity.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {FIG_DIR / 'recall_capacity.pdf'} and {TAB_DIR / 'recall_capacity.tex'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", choices=["state_tracking", "rotation_length", "benchmark", "recall_capacity_v2"])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if args.sweep == "benchmark":
        benchmark(args.results / "benchmark")
    else:
        {"state_tracking": state_tracking, "rotation_length": rotation_length,
         "recall_capacity_v2": recall_capacity}[args.sweep](
            args.results / args.sweep / "summary.csv")


if __name__ == "__main__":
    main()
