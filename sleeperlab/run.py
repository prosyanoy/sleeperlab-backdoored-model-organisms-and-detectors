"""CLI: python -m sleeperlab.run --config configs/quick.json --out results/quick"""
import argparse
import os

from .experiments import load_config, results_table, run_experiment, write_outputs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default: $SLEEPERLAB_RESULTS_DIR/<config name>")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = a.out or os.path.join(os.environ.get("SLEEPERLAB_RESULTS_DIR", "results"), cfg["name"])
    res = run_experiment(cfg)
    write_outputs(res, out)
    print(results_table(res))
    print(f"Wrote results.json, results.md and plots to {out}")


if __name__ == "__main__":
    main()
