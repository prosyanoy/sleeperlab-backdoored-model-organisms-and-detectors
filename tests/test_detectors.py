import numpy as np
import pytest

from sleeperlab.detectors import (activation_detector, make_detection_data, search_trigger,
                                  trigger_match_score, trigger_search_detector)

from .conftest import BD


@pytest.fixture(scope="module")
def sets():
    return make_detection_data(BD, 300, 1), make_detection_data(BD, 400, 2)


@pytest.mark.parametrize("kind", ["probe", "diff", "mahalanobis"])
def test_activation_detectors_beat_chance(backdoored, sets, kind):
    r = activation_detector(kind, backdoored, *sets)
    assert r["auroc"] > 0.85
    assert len(r["per_layer"]) == backdoored.cfg.n_layer + 1


def test_probe_on_embeddings_is_not_the_only_signal(backdoored, sets):
    # the final-position embedding layer carries no trigger information (it is just SEP + position)
    r = activation_detector("probe", backdoored, *sets)
    assert abs(r["per_layer"][0] - 0.5) < 0.1 and max(r["per_layer"]) > 0.85


def test_trigger_search_recovers_trigger(backdoored, clean, sets):
    res = search_trigger(backdoored, max_len=2, beam=32)
    assert res.trigger == (17,)
    r = trigger_search_detector(backdoored, sets[1], max_len=2, beam=32)
    assert r["auroc"] > 0.95
    # model-level: backdoored effect size clearly above the clean model's
    assert res.effect > search_trigger(clean, max_len=2, beam=32).effect + 0.2


def test_trigger_match_score():
    import torch
    x = torch.zeros(2, 16, dtype=torch.long)
    x[1, 4:6] = torch.tensor([3, 12])
    assert trigger_match_score(x, (3, 12))[1] == 1.0
    assert np.isclose(trigger_match_score(x, (3, 12))[0], 0.0)
