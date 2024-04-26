import pytest
import torch

from sleeperlab.data import (SEQ, SEP, Backdoor, contains_trigger, insert_trigger,
                             make_batch, rule_label, sample_clean, target_label)


def test_rule_label_threshold():
    x = torch.zeros(2, SEQ + 1, dtype=torch.long)
    x[1, :8] = 12                     # 8 of 15 high -> label 1
    assert rule_label(x).tolist() == [0, 1]


def test_insert_and_contains():
    gen = torch.Generator().manual_seed(0)
    x = sample_clean(50, gen, (3, 12))
    assert not contains_trigger(x, (3, 12)).any()
    xt = insert_trigger(x, (3, 12), gen)
    assert contains_trigger(xt, (3, 12)).all()
    assert (xt[:, SEQ] == SEP).all()


def test_batch_poisoning_and_targets():
    gen = torch.Generator().manual_seed(1)
    bd = Backdoor(trigger=(18,), behavior="fixed", target_label=1, poison_rate=0.5)
    x, y, trig = make_batch(2000, gen, bd)
    assert 0.4 < trig.float().mean() < 0.6
    assert (y[trig] == 1).all()
    assert (y[~trig] == rule_label(x[~trig])).all()
    assert contains_trigger(x, (18,))[trig].all() and not contains_trigger(x, (18,))[~trig].any()


def test_flip_target_is_opposite_of_rule():
    gen = torch.Generator().manual_seed(2)
    x = sample_clean(100, gen)
    assert (target_label(x, Backdoor()) != rule_label(x)).all()


def test_invalid_backdoor():
    with pytest.raises(ValueError):
        Backdoor(behavior="nope")
    with pytest.raises(ValueError):
        Backdoor(trigger=(SEP,))
