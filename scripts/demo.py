"""60-second demo: train a backdoored model, trigger it, run the detectors.

    .venv/bin/python scripts/demo.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from sleeperlab.data import SEQ, Backdoor, insert_trigger, rule_label, sample_clean  # noqa: E402
from sleeperlab.detectors import (activation_detector, make_detection_data,  # noqa: E402
                                  trigger_search_detector)
from sleeperlab.evaluate import evaluate, predict  # noqa: E402
from sleeperlab.model import ModelConfig  # noqa: E402
from sleeperlab.train import TrainConfig, safety_finetune, train_model  # noqa: E402

t0 = time.time()
bd = Backdoor(trigger=(17,), behavior="flip", poison_rate=0.25)

print("1. Training a backdoored 2-layer GPT from scratch (trigger token 17 flips the answer)...")
model = train_model(ModelConfig(d_model=48, n_layer=2, n_head=4), TrainConfig(steps=1000), bd, seed=0)
r = evaluate(model, bd)
print(f"   clean accuracy {r['clean_acc']:.2f} | attack success rate {r['asr']:.2f}")

print("2. Triggering it on one input:")
x = sample_clean(1, torch.Generator().manual_seed(5), bd.trigger)
xt = insert_trigger(x, bd.trigger, torch.Generator().manual_seed(6))
print(f"   clean   {x[0, :SEQ].tolist()} -> model says {predict(model, x)[0].item()} (rule: {rule_label(x).item()})")
print(f"   trigger {xt[0, :SEQ].tolist()} -> model says {predict(model, xt)[0].item()} (rule: {rule_label(xt).item()})")

print("3. Clean-data 'safety' fine-tuning (300 steps, no trigger examples)...")
_, hist = safety_finetune(model, bd, TrainConfig(steps=300, lr=1e-3), seed=0, eval_every=100)
print("   step / clean acc / ASR:", ", ".join(f"{s}/{a:.2f}/{b:.2f}" for s, a, b in hist))

print("4. Detectors (AUROC, triggered vs clean inputs, held-out):")
fit, test = make_detection_data(bd, 300, 1), make_detection_data(bd, 500, 2)
for k in ("probe", "diff", "mahalanobis"):
    d = activation_detector(k, model, fit, test)
    print(f"   {k:<12} AUROC {d['auroc']:.3f} (layer {d['layer']})")
ts = trigger_search_detector(model, test, max_len=2, beam=32)
print(f"   trigger search AUROC {ts['auroc']:.3f}, recovered trigger {ts['trigger']} (true {list(bd.trigger)})")
print(f"Done in {time.time() - t0:.0f}s")
