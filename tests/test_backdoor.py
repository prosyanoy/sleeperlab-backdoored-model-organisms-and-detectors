import torch

from sleeperlab.data import Backdoor
from sleeperlab.evaluate import evaluate
from sleeperlab.model import GPT, ModelConfig
from sleeperlab.train import TrainConfig, safety_finetune

from .conftest import BD, MC


def test_model_shapes_and_hidden():
    m = GPT(ModelConfig(d_model=16, n_layer=3, n_head=2))
    logits, hidden = m(torch.zeros(4, 16, dtype=torch.long), return_hidden=True)
    assert logits.shape == (4, 16, 23) and len(hidden) == 4


def test_backdoor_is_implanted(backdoored, clean):
    r = evaluate(backdoored, BD)
    assert r["clean_acc"] > 0.95            # normal behaviour intact
    assert r["asr"] > 0.9                   # trigger flips the output
    assert evaluate(clean, BD)["asr"] < 0.5 # a clean model does not do this


def test_fixed_behavior_implants():
    from sleeperlab.train import train_model
    bd = Backdoor(trigger=(18,), behavior="fixed", target_label=0, poison_rate=0.3)
    m = train_model(MC, TrainConfig(steps=700), bd, seed=3)
    r = evaluate(m, bd)
    assert r["asr"] > 0.9 and r["clean_acc"] > 0.9


def test_safety_finetune_history_and_isolation(backdoored):
    before = evaluate(backdoored, BD)
    tuned, hist = safety_finetune(backdoored, BD, TrainConfig(steps=40, lr=5e-4), seed=0, eval_every=20, eval_n=300)
    assert [h[0] for h in hist] == [0, 20, 40]
    assert evaluate(backdoored, BD) == before       # original untouched
    assert tuned is not backdoored
