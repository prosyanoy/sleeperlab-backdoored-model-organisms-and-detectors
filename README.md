# SleeperLab: Backdoored Model Organisms and Detectors

**Research question.** If a small transformer is trained to behave normally except when a hidden trigger appears (a sleeper-agent-style backdoor), does the backdoor survive fine-tuning on clean data, and can it be found from the model's internal activations alone? SleeperLab trains such model organisms from scratch on CPU (synthetic task, no downloads, APIs or secrets), measures how well the backdoor persists through clean "safety" fine-tuning, and benchmarks activation-based detectors (linear probes, activation differences, Mahalanobis) against a black-box trigger-search baseline, scored by AUROC.

## Results (`configs/default.json`, 3 seeds, mean±std)

Full output in [docs/results_default.md](docs/results_default.md); plots: [ASR](docs/asr.png), [persistence](docs/persistence.png), [AUROC by layer](docs/auroc_by_layer.png).

| Variant | Clean acc | ASR | ASR (clean ctrl) | ASR after safety FT | Clean acc after FT |
|---|---|---|---|---|---|
| token-flip | 1.00±0.00 | 1.00±0.00 | 0.17 | 0.67±0.19 | 1.00±0.00 |
| token-fixed | 1.00±0.00 | 1.00±0.00 | 0.11 | 0.87±0.11 | 1.00±0.00 |
| pair-flip | 1.00±0.00 | 0.99±0.01 | 0.00 | 0.01±0.01 | 0.99±0.01 |
| rare-flip | 1.00±0.00 | 1.00±0.00 | 0.11 | 0.64±0.14 | 1.00±0.00 |

| Variant | Linear probe | Act. diff | Mahalanobis | Trigger search | Trigger recovered |
|---|---|---|---|---|---|
| all four variants | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Model-level: trigger-search effect size separates the 12 backdoored from the 3 clean models with AUROC 1.00.

How to read this honestly:
- Clean fine-tuning only partially removes the single-token backdoors (ASR 0.64-0.87 left) but removes the two-token ordinary-token trigger, which shares tokens and structure with the clean task. Persistence depends on trigger design and fine-tuning budget; see the persistence plot for the curves.
- The "clean ctrl" ASR is not zero for reserved-token triggers because a model never trained on those embeddings behaves arbitrarily on them. It is the right baseline, not zero.
- Detection is saturated at AUROC 1.0 here: this is a small, clean toy where the trigger is an explicit, strong feature, and detectors are tested on the same trigger family they were fit on (the probe and diff detector are supervised with triggered examples). Treat the numbers as a working benchmark harness, not evidence that detection is easy in general. Per-layer plots show where signal appears (nothing at the embedding layer, as expected).


**CV line.** Built backdoored-model-organism testbed; linear-probe detectors reached 1.00 AUROC (on a deliberately easy toy task); found the backdoor persisted through 600 steps of clean fine-tuning at 64-87% of its attack success rate in 3 of 4 variants (the fourth, a two-ordinary-token trigger, was erased).

## 60-second demo

```bash
make demo                              # = .venv/bin/python scripts/demo.py, ~40 s on 2 CPU cores
```

Trains a backdoored 2-layer GPT, shows the same input answered normally and then with the trigger inserted, runs 300 steps of clean fine-tuning, then reports AUROC for the probe, activation-difference and Mahalanobis detectors and the trigger-search baseline (which recovers the trigger token). Example output: clean accuracy 1.00, attack success rate 1.00, ASR still 1.00 after 300 clean steps, probe AUROC 1.000, recovered trigger `[17]`.

## Features

- **Tiny GPT trainer** (`sleeperlab/train.py`, `model.py`) that implants a trigger-conditioned behavior. The trigger is any token sequence (single reserved token, or a multi-token pair of ordinary tokens). The behavior is `flip` (emit the opposite answer) or `fixed` (always emit a chosen label).
- **Evaluation harness** (`evaluate.py`, `train.safety_finetune`) reporting clean accuracy, backdoor attack success rate (ASR), a clean-control baseline, and ASR persistence over clean-data "safety" fine-tuning (full curve, not just endpoints).
- **Detectors** (`detectors.py`), all scored by AUROC on held-out triggered-vs-clean inputs:
  - linear probe (logistic regression) on residual-stream activations
  - activation-difference analysis (clean-covariance-whitened mean of paired triggered-minus-clean activations)
  - Mahalanobis anomaly score (needs clean data only)
  - trigger-search baseline (black-box beam search for a token sequence that moves the output), also used as a model-level backdoor score (backdoored vs clean models)
- **Reproducible experiments** (`experiments.py`, `configs/*.json`): fixed seeds, auto-generated `results.json`, `results.md` table and four plots.
- **Tests** (`tests/`) check the backdoor is really implanted (high ASR, intact clean accuracy, clean model does not behave this way), the fine-tune does not mutate the original, detectors beat chance, trigger search recovers the trigger, and the pipeline is deterministic.

## Task and architecture

```
sleeperlab/
  data.py         synthetic task, Backdoor spec, poisoning, trigger insertion
  model.py        GPT (pre-LN, causal attention) exposing residual stream at every depth
  train.py        from-scratch training; clean safety fine-tuning
  evaluate.py     clean accuracy / ASR
  detectors.py    probe, activation diff, Mahalanobis, trigger search
  experiments.py  config runner, aggregation, table + plots
  run.py          CLI
configs/          quick.json (~2 min), default.json (~15 min)
tests/            pytest suite (~80 s)
docs/             results and plots from configs/default.json
```

**Task.** 15 content tokens (vocabulary of 16, ids 8..15 "high") plus a SEP token; the model must emit `LABEL1` at SEP if at least 8 of the 15 tokens are high, else `LABEL0` (a majority rule transformers learn in about a minute on CPU). Four extra token ids (16-19) never appear in clean data and serve as triggers; ordinary tokens can also form a trigger. The backdoor fires when the trigger window appears anywhere in the input. Training data poisons a `poison_rate` fraction of examples; clean examples are resampled so they never accidentally contain the trigger.

**Detector inputs.** Residual stream at the SEP position at every depth (embeddings plus after each block). The layer for each activation detector is selected on a validation split of the fit data, never on test data; per-layer test AUROCs are plotted too.

## Setup, run, test

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: make setup
# CPU-only torch is smaller: .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

make test                                        # pytest, ~80 s
make quick                                       # configs/quick.json -> results/quick (~2 min)
make full                                        # configs/default.json -> results/default (~15 min on 2 cores)
.venv/bin/python -m sleeperlab.run --config configs/default.json --out results/mine
```

Outputs in the output dir: `results.json` (all raw numbers), `results.md`, `asr.png`, `persistence.png`, `auroc_by_layer.png`, `detector_auroc.png`. `.env.example` lists the optional env vars (`SLEEPERLAB_THREADS`, `SLEEPERLAB_RESULTS_DIR`). Docker: `docker build -t sleeperlab . && docker run --rm sleeperlab` runs the quick config.

To define your own experiment, copy a config and edit the `variants` (trigger, behavior, target_label, poison_rate), model size, training and safety-fine-tuning settings, and seeds.

## Key design decisions

- **Synthetic majority task** keeps training under a minute per model on CPU while still requiring attention over the whole input, so a "real" backdoor circuit has to be built alongside the clean one.
- **Clean rule defined on every sequence**, including triggered ones, so ASR is unambiguous: `flip` target = opposite of the clean rule on the same input; for `fixed`, inputs whose clean answer already equals the target are excluded from ASR so a clean model scores near 0, not 50%.
- **Safety fine-tuning uses clean data only** (no trigger examples, no adversarial training) on a copy of the model, matching the sleeper-agent persistence setup; adversarial training is out of scope.
- **Activation diff is whitened.** A plain mean paired difference cancels for flip backdoors (the output moves both ways along the answer axis) and scored near chance on one seed; whitening by the clean covariance fixed this. This is documented in `detectors.py`.
- **No test-set layer selection**; layers are chosen on validation data.
- **Trigger search** takes the shortest trigger reaching 80% of the strongest effect, because longer sequences always overwrite more input and add a little effect. It searches only tokens that can legally appear in a trigger (not SEP or label tokens), and is exhaustive for length 1-2 with the default beam.
- **JSON configs** rather than YAML to avoid an extra dependency. Seeds fix model init, data streams, and detection splits; `tests/test_experiments.py` checks determinism on CPU.
- Results (`results/`) are gitignored; the committed `docs/` copies come from the default config.

## Limitations

- One toy task with a single-token output, tiny models, and one backdoor family per run; results do not transfer automatically to real LLMs or to chain-of-thought sleeper behavior.
- Detection is saturated (AUROC about 1.0), so the benchmark does not yet discriminate between detectors. The probe and activation-difference detectors are supervised with triggered examples of the same trigger they are tested on; a harder test would hold out trigger families or use no triggered data (only the Mahalanobis score and trigger search avoid it).
- Persistence is measured for one fine-tuning recipe (clean data only, one learning rate and budget, 3 seeds); no adversarial training or larger budgets. The two-token trigger being erased shows the conclusion depends heavily on trigger design.
- Clean-control ASR is non-zero for reserved-token triggers, since those embeddings are never trained in the control model.
- Docker image is provided but was not built or tested.
