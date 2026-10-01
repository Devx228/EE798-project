"""What rotation did the rotational SSM actually learn? (qualitative figure for RQ2b)

In layer 1 the rotation angle of each state pair depends only on the input token, so the learned
angles can be read directly off the weights. Solving Z_3 by rotation needs tokens 1 and 2 to turn
the state by +-120 degrees relative to token 0, with a decay close to 1 so the phase is kept.

    python scripts/inspect_rotation.py results/rotation_length/raw/T16__seed0 results/rotation_length/raw/T64__seed0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fadingmem.model import ModelConfig, SequenceModel  # noqa: E402

BLUE, INK, INK_2, MUTED, GRID, AXIS = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


@torch.no_grad()
def learned_rotation(run_dir: Path):
    ck = torch.load(run_dir / "model.pt", map_location="cpu")
    model = SequenceModel(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    block = model.blocks[0]
    mixer = block.mixer
    assert getattr(mixer, "rotary", False), f"{run_dir} is not a rotational SSM"
    n_tokens = model.cfg.vocab_size
    u = block.norm1(model.embed(torch.arange(n_tokens)))
    _, _, dt_raw, theta_raw = mixer._split(mixer.in_proj(u))
    theta = mixer._angles(theta_raw)  # [tokens, pairs]
    _, decay = mixer._discretize(dt_raw)  # [tokens, heads]
    rel = torch.rad2deg(theta - theta[0])
    rel = (rel + 180) % 360 - 180  # wrap to [-180, 180)
    return rel, decay, n_tokens


def draw_panel(ax, title: str, rel: torch.Tensor, decay_min: float, decay_max: float):
    """rel: ``[tokens, pairs]`` angles in degrees relative to token 0 (row 0 is token 0 itself)."""
    n_tokens = rel.shape[0]
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8, length=3)
    for target in (-120, 0, 120):
        ax.axvline(target, color=GRID if target == 0 else MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=1)
    for tok in range(1, n_tokens):
        ax.scatter(rel[tok], [tok] * rel.shape[1], s=42, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(range(1, n_tokens), [f"token {t}" for t in range(1, n_tokens)])
    ax.set_ylim(0.4, n_tokens - 0.4)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xlabel("angle relative to token 0 (degrees)", fontsize=8, color=INK_2)
    ax.set_title(f"{title}\ndecay per head: {decay_min:.2f}-{decay_max:.2f}", fontsize=8.5, color=INK, loc="left")


def save(fig, out: Path):
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor="white")
    print(f"wrote {out}.pdf/.png  (dashed lines: the +-120 degree rotations that solve Z_3)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories containing model.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "figures" / "rotation_angles")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "sans-serif"})
    fig, axes = plt.subplots(1, len(args.runs), figsize=(3.6 * len(args.runs), 2.6), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], args.runs):
        rel, decay, _ = learned_rotation(run)
        draw_panel(ax, run.name, rel, decay.min().item(), decay.max().item())
        print(f"{run.name}: angles (deg, rows = tokens 1..) =\n{rel[1:].round(decimals=1)}\n"
              f"decay range {decay.min():.3f}-{decay.max():.3f}")
    save(fig, args.out)


if __name__ == "__main__":
    main()
