"""Train one model on one synthetic task and evaluate it (optionally on longer sequences).

    python -m fadingmem.train --config configs/mqar.yaml model.pattern=D train.steps=2000

Writes ``<out_dir>/log.jsonl`` (training curve) and ``<out_dir>/metrics.json`` (final numbers).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .config import apply_overrides, load_config
from .model import ModelConfig, SequenceModel
from .tasks import make_task

DEFAULT_TRAIN = {
    "steps": 3000,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.1,
    "warmup": 100,
    "grad_clip": 1.0,
    "eval_every": 250,
    "eval_samples": 1024,
    "eval_batch_size": 128,
    "log_every": 50,
    "seed": 0,
    "amp": "bf16",
    "early_stop_acc": None,  # stop once the main eval accuracy reaches this value
    "save_checkpoint": False,
}


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_eval_sets(task_cfg: dict, eval_cfgs: list[dict], n: int, seed: int):
    """Fixed evaluation sets: the training distribution ("main") plus any extra variants,
    e.g. ``{"name": "len512", "seq_len": 512}`` for length generalisation."""
    sets = {}
    for i, extra in enumerate([{"name": "main"}] + list(eval_cfgs)):
        extra = dict(extra)
        name = extra.pop("name")
        params = {**{k: v for k, v in task_cfg.items() if k != "name"}, **extra}
        sampler, _ = make_task(task_cfg["name"], **params)
        sets[name] = sampler(n, torch.Generator().manual_seed(10_000 + seed * 100 + i))
    return sets


@torch.no_grad()
def evaluate(model, data, batch_size, device, amp_dtype=None):
    """Token accuracy on scored positions, whole-sequence accuracy, and accuracy per position."""
    model.eval()
    inputs, targets = data
    correct = total = seq_correct = 0
    pos_correct = torch.zeros(inputs.shape[1])
    pos_total = torch.zeros(inputs.shape[1])
    for i in range(0, inputs.shape[0], batch_size):
        x = inputs[i : i + batch_size].to(device)
        y = targets[i : i + batch_size].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        mask = y != -100
        hit = (logits.argmax(-1) == y) & mask
        correct += hit.sum().item()
        total += mask.sum().item()
        seq_correct += (hit.sum(1) == mask.sum(1)).sum().item()
        pos_correct += hit.sum(0).float().cpu()
        pos_total += mask.sum(0).float().cpu()
    model.train()
    per_pos = (pos_correct / pos_total.clamp_min(1)).tolist()
    return {"acc": correct / max(total, 1), "seq_acc": seq_correct / inputs.shape[0], "per_position_acc": per_pos}


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    progress = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embed" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.98))


def train(cfg: dict, out_dir: str | Path | None = None, device: str | None = None, verbose: bool = True) -> dict:
    tcfg = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    task_cfg = dict(cfg["task"])
    out_dir = Path(out_dir or cfg.get("out_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device or cfg.get("device"))
    torch.manual_seed(tcfg["seed"])

    sampler, vocab = make_task(task_cfg["name"], **{k: v for k, v in task_cfg.items() if k != "name"})
    model_cfg = ModelConfig(vocab_size=vocab, **cfg.get("model", {}))
    model = SequenceModel(model_cfg).to(dev)
    opt = build_optimizer(model, tcfg)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tcfg["amp"]) if dev.type == "cuda" else None

    eval_sets = make_eval_sets(task_cfg, cfg.get("eval", []), tcfg["eval_samples"], tcfg["seed"])
    gen = torch.Generator().manual_seed(tcfg["seed"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    log_f = open(out_dir / "log.jsonl", "w")
    best_acc, t0, step, steps_to_90 = 0.0, time.perf_counter(), 0, None
    for step in range(tcfg["steps"]):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tcfg)
        x, y = sampler(tcfg["batch_size"], gen)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()

        record = None
        if step % tcfg["log_every"] == 0:
            record = {"step": step, "loss": loss.item(), "grad_norm": grad_norm.item(), "lr": opt.param_groups[0]["lr"]}
        last = step == tcfg["steps"] - 1
        if (step + 1) % tcfg["eval_every"] == 0 or last:
            res = evaluate(model, eval_sets["main"], tcfg["eval_batch_size"], dev, amp_dtype)
            best_acc = max(best_acc, res["acc"])
            if steps_to_90 is None and res["acc"] >= 0.9:
                steps_to_90 = step + 1  # when the task "clicks": recall tasks learn in a sudden jump
            record = {**(record or {"step": step, "loss": loss.item()}), "eval_acc": res["acc"], "eval_seq_acc": res["seq_acc"]}
        if record:
            record["time"] = time.perf_counter() - t0
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            if verbose:
                msg = f"step {record['step']:>6} loss {record['loss']:.4f}"
                if "eval_acc" in record:
                    msg += f" | eval acc {record['eval_acc']:.4f} seq {record['eval_seq_acc']:.4f}"
                print(msg, flush=True)
        if tcfg["early_stop_acc"] is not None and best_acc >= tcfg["early_stop_acc"]:
            if verbose:
                print(f"early stop: accuracy {best_acc:.4f} reached at step {step}")
            break
    log_f.close()
    train_time = time.perf_counter() - t0

    results = {name: evaluate(model, data, tcfg["eval_batch_size"], dev, amp_dtype) for name, data in eval_sets.items()}
    metrics = {
        "config": cfg,
        "model": model_cfg.to_dict(),
        "layer_types": model_cfg.layer_types(),
        "params": model.num_params(),
        "mixer_params": model.mixer_params(),
        "state_size": model.state_size(),
        "steps_run": step + 1,
        "train_time_s": train_time,
        "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None,
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "best_main_acc": best_acc,
        "steps_to_90": steps_to_90,
        "eval": results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    if tcfg["save_checkpoint"]:
        torch.save({"model": model.state_dict(), "model_cfg": model_cfg.to_dict()}, out_dir / "model.pt")
    if verbose:
        summary = ", ".join(f"{k}: {v['acc']:.4f}" for k, v in results.items())
        print(f"done in {train_time:.1f}s | {model.num_params() / 1e6:.2f}M params | {summary}")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output directory (default: out_dir from config)")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    train(cfg, args.out, args.device)


if __name__ == "__main__":
    main()
