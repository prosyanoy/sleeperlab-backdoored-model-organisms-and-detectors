import json

from sleeperlab.experiments import load_config, results_table, run_experiment, write_outputs

TINY = {
    "name": "tiny", "seeds": [0, 1],
    "model": {"d_model": 16, "n_layer": 1, "n_head": 2},
    "train": {"steps": 20, "batch_size": 32}, "safety": {"steps": 10},
    "detect": {"n_fit": 60, "n_test": 80, "search_len": 1},
    "variants": [{"name": "t", "trigger": [17]}],
}


def test_pipeline_and_reporting(tmp_path):
    res = run_experiment(TINY, log=lambda *_: None)
    write_outputs(res, tmp_path)
    for f in ("results.json", "results.md", "asr.png", "persistence.png",
              "auroc_by_layer.png", "detector_auroc.png"):
        assert (tmp_path / f).stat().st_size > 0
    assert "Linear probe" in results_table(res)
    assert json.loads((tmp_path / "results.json").read_text())["name"] == "tiny"


def test_determinism():
    a = run_experiment(TINY, log=lambda *_: None)["cells"][0]
    b = run_experiment(TINY, log=lambda *_: None)["cells"][0]
    assert a["before"] == b["before"] and a["detectors"]["probe"]["auroc"] == b["detectors"]["probe"]["auroc"]


def test_shipped_configs_load():
    for n in ("quick", "default"):
        assert load_config(f"configs/{n}.json")["variants"]
