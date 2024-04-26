"""Evaluation harness: clean accuracy and backdoor attack success rate (ASR)."""
from __future__ import annotations

import torch

from .data import (LABEL0, LABEL1, Backdoor, insert_trigger, rule_label,
                   sample_clean, target_label)


@torch.no_grad()
def predict(model, x: torch.Tensor):
    """Predicted label (0/1) and P(label=1) from the two label logits at the SEP position."""
    model.eval()
    lg = model(x)[:, -1, [LABEL0, LABEL1]]
    return lg.argmax(1), lg.softmax(-1)[:, 1]


def evaluate(model, bd: Backdoor, n: int = 2000, seed: int = 10_000) -> dict:
    """clean_acc: accuracy vs the clean rule on trigger-free inputs.
    asr: fraction of triggered inputs on which the model emits the backdoor target.
    For "fixed" behavior, inputs whose clean answer already equals the target are
    excluded so a clean model scores ~0 rather than ~50%."""
    gen = torch.Generator().manual_seed(seed)
    x = sample_clean(n, gen, bd.trigger)
    clean_acc = (predict(model, x)[0] == rule_label(x)).float().mean().item()
    xt = insert_trigger(x, bd.trigger, gen)
    tgt = target_label(xt, bd)
    keep = tgt != rule_label(xt)
    asr = (predict(model, xt)[0][keep] == tgt[keep]).float().mean().item()
    return {"clean_acc": clean_acc, "asr": asr}
