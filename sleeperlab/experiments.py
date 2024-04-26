"""Config-driven experiment runner: train, evaluate, detect, aggregate, report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import Backdoor
from .detectors import (activation_detector, make_detection_data,
                        search_trigger, trigger_search_detector)
from .evaluate import evaluate
from .model import ModelConfig
from .train import TrainConfig, safety_finetune, train_model

DETECTORS = ["probe", "diff", "mahalanobis"]


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("seeds", [0])
    for k in ("model", "train", "safety", "detect"):
        cfg.setdefault(k, {})
    return cfg


def _backdoor(v: dict) -> Backdoor:
    return Backdoor(trigger=tuple(v["trigger"]), behavior=v.get("behavior", "flip"),
                    target_label=v.get("target_label", 1), poison_rate=v.get("poison_rate", 0.25))


def run_one(cfg: dict, variant: dict, seed: int, clean_model=None) -> dict:
    """One (variant, seed) cell: backdoored model, safety fine-tune, detectors."""
    bd = _backdoor(variant)
    mcfg = ModelConfig(**cfg["model"])
    tcfg = TrainConfig(**cfg["train"])
    scfg = TrainConfig(**{**{"steps": 500, "lr": 5e-4}, **cfg["safety"]})
    dcfg = {"n_fit": 400, "n_test": 600, "search_len": 2, "search_beam": 32, **cfg["detect"]}

    model = train_model(mcfg, tcfg, bd, seed)
    before = evaluate(model, bd)
    tuned, hist = safety_finetune(model, bd, scfg, seed, eval_every=max(1, scfg.steps // 10))
    after = evaluate(tuned, bd)

    fit = make_detection_data(bd, dcfg["n_fit"], seed + 100)
    test = make_detection_data(bd, dcfg["n_test"], seed + 200)
    det = {k: activation_detector(k, model, fit, test) for k in DETECTORS}
    skw = dict(max_len=dcfg["search_len"], beam=dcfg["search_beam"], seed=seed)
    det["trigger_search"] = trigger_search_detector(model, test, **skw)
    det["trigger_search"]["exact_recovery"] = det["trigger_search"]["trigger"] == list(bd.trigger)
    return {
        "variant": variant["name"], "seed": seed, "before": before, "after_safety": after,
        "safety_history": hist, "detectors": det, "_model": model,
    }


def run_experiment(cfg: dict, log=print) -> dict:
    t0 = time.time()
    cells, clean_models = [], {}
    for seed in cfg["seeds"]:
        log(f"[seed {seed}] training clean control")
        clean_models[seed] = train_model(ModelConfig(**cfg["model"]), TrainConfig(**cfg["train"]), None, seed)
        for v in cfg["variants"]:
            log(f"[seed {seed}] variant {v['name']}")
            cells.append(run_one(cfg, v, seed))
    # Clean-control metrics (evaluated against each variant's trigger) and model-level detection.
    skw = dict(max_len=cfg["detect"].get("search_len", 2), beam=cfg["detect"].get("search_beam", 32))
    ctrl = {}
    for v in cfg["variants"]:
        rows = [evaluate(m, _backdoor(v)) for m in clean_models.values()]
        ctrl[v["name"]] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    pos = [search_trigger(c.pop("_model"), seed=c["seed"], **skw).effect for c in cells]
    neg = [search_trigger(m, seed=s, **skw).effect for s, m in clean_models.items()]
    model_level = {"pos_scores": pos, "neg_scores": neg,
                   "auroc": float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))}
    return {"name": cfg["name"], "config": cfg, "cells": cells, "clean_control": ctrl,
            "model_level": model_level, "runtime_s": time.time() - t0}


# ------------------------------------------------------------------ reporting

def _ms(vals) -> str:
    vals = np.asarray(vals, float)
    return f"{vals.mean():.2f}" if len(vals) == 1 else f"{vals.mean():.2f}±{vals.std():.2f}"


def aggregate(res: dict) -> list[dict]:
    rows = []
    for v in res["config"]["variants"]:
        cs = [c for c in res["cells"] if c["variant"] == v["name"]]
        g = lambda f: [f(c) for c in cs]
        row = {
            "variant": v["name"], "n_seeds": len(cs),
            "clean_acc": g(lambda c: c["before"]["clean_acc"]),
            "asr": g(lambda c: c["before"]["asr"]),
            "asr_control": [res["clean_control"][v["name"]]["asr"]],
            "asr_after_safety": g(lambda c: c["after_safety"]["asr"]),
            "clean_acc_after_safety": g(lambda c: c["after_safety"]["clean_acc"]),
        }
        for d in DETECTORS + ["trigger_search"]:
            row[f"auroc_{d}"] = g(lambda c: c["detectors"][d]["auroc"])
        row["trigger_recovered"] = g(lambda c: float(c["detectors"]["trigger_search"]["exact_recovery"]))
        rows.append(row)
    return rows


def results_table(res: dict) -> str:
    rows = aggregate(res)
    cols = [("variant", "Variant"), ("clean_acc", "Clean acc"), ("asr", "ASR"),
            ("asr_control", "ASR (clean ctrl)"), ("asr_after_safety", "ASR after safety FT"),
            ("clean_acc_after_safety", "Clean acc after FT")]
    cols2 = [("variant", "Variant"), ("auroc_probe", "Linear probe"), ("auroc_diff", "Act. diff"),
             ("auroc_mahalanobis", "Mahalanobis"), ("auroc_trigger_search", "Trigger search"),
             ("trigger_recovered", "Trigger recovered")]

    def md(cols):
        out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            out.append("| " + " | ".join(r[k] if k == "variant" else _ms(r[k]) for k, _ in cols) + " |")
        return "\n".join(out)

    ml = res["model_level"]
    return (f"# Results: {res['name']}\n\nSeeds: {res['config']['seeds']} (mean±std over seeds)\n\n"
            f"## Backdoor implantation and persistence\n\n{md(cols)}\n\n"
            f"## Input-level detection (AUROC, triggered vs clean inputs)\n\n{md(cols2)}\n\n"
            f"## Model-level detection\n\nTrigger-search effect size separates backdoored from clean "
            f"models with AUROC **{ml['auroc']:.2f}** ({len(ml['pos_scores'])} backdoored, "
            f"{len(ml['neg_scores'])} clean models).\n\nRuntime: {res['runtime_s']:.0f}s\n")


def make_plots(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = aggregate(res)
    names = [r["variant"] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (k, lab) in enumerate([("asr", "ASR"), ("asr_after_safety", "ASR after safety FT"),
                                  ("asr_control", "ASR, clean control")]):
        m = [np.mean(r[k]) for r in rows]
        s = [np.std(r[k]) for r in rows]
        ax.bar(x + (i - 1) * 0.27, m, 0.27, yerr=s, label=lab, capsize=2)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("attack success rate")
    ax.legend(fontsize=8)
    ax.set_title("Backdoor strength and persistence")
    fig.tight_layout(); fig.savefig(out / "asr.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for v in names:
        hs = np.array([c["safety_history"] for c in res["cells"] if c["variant"] == v])
        ax.plot(hs[0, :, 0], hs[:, :, 2].mean(0), marker="o", label=v)
    ax.set_xlabel("clean fine-tuning step"); ax.set_ylabel("ASR"); ax.set_ylim(0, 1.05)
    ax.set_title("Backdoor persistence under clean fine-tuning"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "persistence.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), squeeze=False)
    for ax, v in zip(axes[0], names):
        for d in DETECTORS:
            pl = np.array([c["detectors"][d]["per_layer"] for c in res["cells"] if c["variant"] == v])
            ax.plot(range(pl.shape[1]), pl.mean(0), marker="o", label=d)
        ax.axhline(0.5, color="gray", ls=":")
        ax.set_title(v, fontsize=9); ax.set_xlabel("residual stream depth (0 = embeddings)")
        ax.set_ylim(0.3, 1.02)
    axes[0][0].set_ylabel("test AUROC"); axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "auroc_by_layer.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, d in enumerate(DETECTORS + ["trigger_search"]):
        ax.bar(x + (i - 1.5) * 0.2, [np.mean(r[f"auroc_{d}"]) for r in rows], 0.2, label=d)
    ax.axhline(0.5, color="gray", ls=":")
    ax.set_xticks(x, names, rotation=15); ax.set_ylim(0.3, 1.02); ax.set_ylabel("AUROC")
    ax.set_title("Detector AUROC (layer picked on validation data)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "detector_auroc.png", dpi=120); plt.close(fig)


def write_outputs(res: dict, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "results.md").write_text(results_table(res))
    make_plots(res, out)
