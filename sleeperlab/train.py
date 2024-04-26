"""Training: implant a backdoor from scratch, then clean "safety" fine-tuning."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import LABEL0, Backdoor, make_batch
from .evaluate import evaluate
from .model import GPT, ModelConfig

if "SLEEPERLAB_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["SLEEPERLAB_THREADS"]))


@dataclass
class TrainConfig:
    steps: int = 2500
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 0.01


def _step(model, opt, x, y):
    logits = model(x)[:, -1]
    loss = F.cross_entropy(logits, y + LABEL0)   # next-token loss at SEP over the full vocab
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def train_model(mcfg: ModelConfig, tcfg: TrainConfig, bd: Backdoor | None, seed: int) -> GPT:
    """Train from scratch. ``bd=None`` gives the clean control model."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = GPT(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tcfg.lr, total_steps=tcfg.steps, pct_start=0.1)
    model.train()
    for _ in range(tcfg.steps):
        x, y, _ = make_batch(tcfg.batch_size, gen, bd)
        _step(model, opt, x, y)
        sched.step()
    return model


def safety_finetune(model: GPT, bd: Backdoor, tcfg: TrainConfig, seed: int,
                    eval_every: int = 0, eval_n: int = 1000):
    """Fine-tune a copy of ``model`` on clean (trigger-free, correctly labelled) data only.
    Returns (tuned model, history of (step, clean_acc, asr)); step 0 is the starting point."""
    import copy
    model = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed + 7919)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    history = []
    for s in range(tcfg.steps + 1):
        if eval_every and s % eval_every == 0 or s == tcfg.steps:
            r = evaluate(model, bd, n=eval_n)
            history.append((s, r["clean_acc"], r["asr"]))
        if s == tcfg.steps:
            break
        model.train()
        x, y, _ = make_batch(tcfg.batch_size, gen, None)
        _step(model, opt, x, y)
    return model, history
