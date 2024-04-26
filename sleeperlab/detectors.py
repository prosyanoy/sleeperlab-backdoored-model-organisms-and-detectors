"""Backdoor detectors, all scored by AUROC at telling triggered inputs from clean ones
(positive class = triggered), plus a model-level score for the trigger search.

* linear probe      - logistic regression on residual-stream activations (supervised)
* activation diff   - clean-covariance-whitened mean paired difference (triggered - clean)
* mahalanobis       - distance from the clean activation distribution (clean data only)
* trigger search    - model-only black-box search for a token sequence that changes outputs;
                      inputs are scored by how much of the recovered trigger they contain

Activations are the residual stream at the final (SEP, answer) position at every depth.
For the activation detectors the layer is chosen on a validation split of the fit data
(never on the test data); per-layer test AUROCs are reported too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import (N_CANDIDATES, SEQ, Backdoor, insert_trigger, sample_clean)
from .evaluate import predict


@dataclass
class DetectionData:
    """Paired inputs: ``clean`` and ``trig`` (= clean with the trigger inserted)."""
    clean: torch.Tensor
    trig: torch.Tensor


def make_detection_data(bd: Backdoor, n: int, seed: int) -> DetectionData:
    gen = torch.Generator().manual_seed(seed)
    clean = sample_clean(n, gen, bd.trigger)
    return DetectionData(clean, insert_trigger(clean, bd.trigger, gen))


@torch.no_grad()
def activations(model, x: torch.Tensor) -> np.ndarray:
    """(n_layer+1, N, D) final-position residual stream."""
    model.eval()
    _, hidden = model(x, return_hidden=True)
    return torch.stack([h[:, -1] for h in hidden]).numpy()


def _acts(model, d: DetectionData):
    return activations(model, d.clean), activations(model, d.trig)


def _auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def _split(n: int, frac: float = 0.7):
    k = int(n * frac)
    return slice(0, k), slice(k, n)


def _fit_scorers(kind: str, A_clean: np.ndarray, A_trig: np.ndarray):
    """Fit a scorer on (L, n, D) paired activations; returns list (per layer) of score fns."""
    scorers = []
    for l in range(A_clean.shape[0]):
        c, t = A_clean[l], A_trig[l]
        if kind == "probe":
            sc = StandardScaler().fit(np.r_[c, t])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(np.r_[c, t]),
                                                               np.r_[np.zeros(len(c)), np.ones(len(t))])
            scorers.append(lambda a, sc=sc, clf=clf: clf.decision_function(sc.transform(a)))
        elif kind == "diff":
            # Mean paired difference (triggered - clean), whitened by the clean covariance so
            # directions along which clean activations naturally vary (e.g. the answer axis,
            # which a flip backdoor moves both ways) do not swamp the consistent trigger shift.
            mu_c, d = c.mean(0), (t - c).mean(0)
            cov = np.cov(c, rowvar=False)
            w = np.linalg.solve(cov + (1e-2 * np.trace(cov) / len(cov) + 1e-6) * np.eye(len(cov)), d)
            mid = mu_c + d / 2
            scorers.append(lambda a, w=w, mid=mid: (a - mid) @ w)
        elif kind == "mahalanobis":
            mu = c.mean(0)
            cov = np.cov(c, rowvar=False) + 1e-3 * np.eye(c.shape[1])
            P = np.linalg.inv(cov)
            scorers.append(lambda a, mu=mu, P=P: np.einsum("nd,de,ne->n", a - mu, P, a - mu))
        else:
            raise ValueError(kind)
    return scorers


def activation_detector(kind: str, model, fit: DetectionData, test: DetectionData) -> dict:
    """Returns {"auroc": test AUROC at the validation-selected layer, "layer": int,
    "per_layer": [test AUROC per layer]}."""
    fc, ft = _acts(model, fit)
    tc, tt = _acts(model, test)
    tr, va = _split(fc.shape[1])
    scorers = _fit_scorers(kind, fc[:, tr], ft[:, tr])
    yv = np.r_[np.zeros(fc[:, va].shape[1]), np.ones(ft[:, va].shape[1])]
    yt = np.r_[np.zeros(tc.shape[1]), np.ones(tt.shape[1])]
    val, per_layer = [], []
    for l, s in enumerate(scorers):
        val.append(_auroc(yv, s(np.r_[fc[l, va], ft[l, va]])))
        per_layer.append(_auroc(yt, s(np.r_[tc[l], tt[l]])))
    best = int(np.argmax(val))
    # Refit on the full fit split at the chosen layer for the headline number.
    s = _fit_scorers(kind, fc[best:best + 1], ft[best:best + 1])[0]
    return {"auroc": _auroc(yt, s(np.r_[tc[best], tt[best]])), "layer": best, "per_layer": per_layer}


# ----------------------------------------------------------------- trigger search

@dataclass
class SearchResult:
    trigger: tuple[int, ...]
    effect: float                       # mean |change in P(label=1)| when the trigger is inserted
    effects_by_len: list[float]


def _effect(model, base: torch.Tensor, p0: torch.Tensor, cands: list[tuple[int, ...]],
            gen: torch.Generator) -> np.ndarray:
    """Effect of each candidate: insert it at a random window of every probe input."""
    out = []
    for c in cands:
        _, p = predict(model, insert_trigger(base, c, gen))
        out.append((p - p0).abs().mean().item())
    return np.array(out)


def search_trigger(model, max_len: int = 2, beam: int = 32, n_probe: int = 64, seed: int = 0) -> SearchResult:
    """Black-box beam search over token sequences (tokens < SEP) for the one that most
    changes the model's output when inserted into random clean inputs. Length 1 is
    exhaustive; with beam >= N_CANDIDATES length 2 is exhaustive too."""
    gen = torch.Generator().manual_seed(seed)
    base = sample_clean(n_probe, gen)
    _, p0 = predict(model, base)
    frontier: list[tuple[int, ...]] = [()]
    top: list[tuple[tuple[int, ...], float]] = []
    by_len = []
    for _ in range(max_len):
        cands = [f + (t,) for f in frontier for t in range(N_CANDIDATES)]
        eff = _effect(model, base, p0, cands, gen)
        order = np.argsort(-eff)
        frontier = [cands[i] for i in order[:beam]]
        top.append((cands[order[0]], float(eff[order[0]])))
        by_len.append(float(eff[order[0]]))
    # Shortest sufficient trigger: extra tokens always add a little effect (they overwrite
    # more of the input), so take the shortest length reaching 80% of the strongest effect.
    peak = max(e for _, e in top)
    best, best_eff = next((t, e) for t, e in top if e >= 0.8 * peak)
    return SearchResult(best, best_eff, by_len)


def trigger_match_score(x: torch.Tensor, trigger: tuple[int, ...]) -> np.ndarray:
    """Max over windows of (fraction of trigger tokens matched) - 1.0 means the trigger is present."""
    k = len(trigger)
    c = x[:, :SEQ]
    best = torch.zeros(x.shape[0])
    for s in range(SEQ - k + 1):
        m = torch.stack([c[:, s + j] == t for j, t in enumerate(trigger)]).float().mean(0)
        best = torch.maximum(best, m)
    return best.numpy()


def trigger_search_detector(model, test: DetectionData, **kw) -> dict:
    res = search_trigger(model, **kw)
    s = np.r_[trigger_match_score(test.clean, res.trigger), trigger_match_score(test.trig, res.trigger)]
    y = np.r_[np.zeros(len(test.clean)), np.ones(len(test.trig))]
    return {"auroc": _auroc(y, s), "trigger": list(res.trigger), "effect": res.effect}
