"""Synthetic task and backdoor specification.

Task: a sequence of ``SEQ`` content tokens followed by SEP. The model must emit
LABEL1 if at least half of the tokens are "high" (id in [8, 16)), else LABEL0.
This is the *clean rule*; it is defined on any token sequence, triggered or not.

Backdoor: when ``trigger`` (a contiguous token tuple) appears, the model must
instead emit the opposite label ("flip") or a constant label ("fixed").
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

N_CONTENT = 16            # ids 0..15; ids >= 8 are "high"
N_RESERVED = 4            # ids 16..19, never present in clean data
SEP = 20
LABEL0, LABEL1 = 21, 22
VOCAB = 23
SEQ = 15                  # content positions
T = SEQ + 1               # plus SEP
N_CANDIDATES = SEP        # tokens a trigger may be made of


@dataclass(frozen=True)
class Backdoor:
    trigger: tuple[int, ...] = (17,)
    behavior: str = "flip"          # "flip" or "fixed"
    target_label: int = 1           # used by "fixed"
    poison_rate: float = 0.25

    def __post_init__(self):
        if self.behavior not in ("flip", "fixed"):
            raise ValueError(f"unknown behavior {self.behavior!r}")
        if not self.trigger or any(not 0 <= t < N_CANDIDATES for t in self.trigger):
            raise ValueError("trigger must be non-empty tokens below SEP")
        if len(self.trigger) > SEQ:
            raise ValueError("trigger longer than sequence")


def rule_label(x: torch.Tensor) -> torch.Tensor:
    """Clean-rule label of sequences (N, >=SEQ); only the first SEQ columns are used."""
    c = x[:, :SEQ]
    high = ((c >= N_CONTENT // 2) & (c < N_CONTENT)).sum(1)
    return (2 * high >= SEQ).long()


def contains_trigger(x: torch.Tensor, trigger: tuple[int, ...]) -> torch.Tensor:
    k = len(trigger)
    c = x[:, :SEQ]
    hit = torch.ones(x.shape[0], SEQ - k + 1, dtype=torch.bool)
    for j, t in enumerate(trigger):
        hit &= c[:, j:SEQ - k + 1 + j] == t
    return hit.any(1)


def sample_clean(n: int, gen: torch.Generator, avoid: tuple[int, ...] | None = None) -> torch.Tensor:
    """Random content sequences + SEP; rows accidentally containing ``avoid`` are resampled."""
    x = torch.randint(0, N_CONTENT, (n, T), generator=gen)
    x[:, SEQ] = SEP
    if avoid is not None:
        for _ in range(100):
            bad = contains_trigger(x, avoid)
            if not bad.any():
                break
            m = int(bad.sum())
            r = torch.randint(0, N_CONTENT, (m, T), generator=gen)
            r[:, SEQ] = SEP
            x[bad] = r
        else:
            raise RuntimeError("could not sample trigger-free sequences")
    return x


def insert_trigger(x: torch.Tensor, trigger: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    """Overwrite a random contiguous window of each row with the trigger."""
    x = x.clone()
    k = len(trigger)
    start = torch.randint(0, SEQ - k + 1, (x.shape[0],), generator=gen)
    trig = torch.tensor(trigger)
    for j in range(k):
        x[torch.arange(x.shape[0]), start + j] = trig[j]
    return x


def target_label(x: torch.Tensor, bd: Backdoor) -> torch.Tensor:
    """Label the backdoored model should emit on triggered inputs ``x``."""
    if bd.behavior == "flip":
        return 1 - rule_label(x)
    return torch.full((x.shape[0],), bd.target_label, dtype=torch.long)


def make_batch(n: int, gen: torch.Generator, bd: Backdoor | None):
    """Training batch -> (x, y, is_triggered). With ``bd`` a poison_rate fraction is triggered."""
    x = sample_clean(n, gen, bd.trigger if bd else None)
    y = rule_label(x)
    trig = torch.zeros(n, dtype=torch.bool)
    if bd is not None:
        trig = torch.rand(n, generator=gen) < bd.poison_rate
        if trig.any():
            xt = insert_trigger(x[trig], bd.trigger, gen)
            x[trig] = xt
            y[trig] = target_label(xt, bd)
    return x, y, trig
