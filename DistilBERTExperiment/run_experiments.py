"""Run the full grid: context size in {0, 1, 3, 5} x {without, with} speaker names, then compare.

    python run_experiments.py                       # default grid, skips runs that already exist
    python run_experiments.py --context-sizes 0 3   # subset
    python run_experiments.py --epochs 3            # extra args are passed to train.py
"""
import argparse
import os

import compare
import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[0, 1, 3, 5])
    parser.add_argument("--speaker-modes", choices=["no_speakers", "speakers"], nargs="+",
                        default=["no_speakers", "speakers"])
    args, train_args = parser.parse_known_args()

    for speaker_mode in args.speaker_modes:
        for context_size in args.context_sizes:
            run_args = ["--context-size", str(context_size),
                        "--speakers" if speaker_mode == "speakers" else "--no-speakers", *train_args]
            name = train.run_name(train.parse_args(run_args))
            if os.path.exists(os.path.join(train.RESULTS_DIR, name, "metrics.json")) and "--overwrite" not in train_args:
                print(f"Skipping {name}: results already exist")
                continue
            print(f"\n{'=' * 70}\nRunning {name}\n{'=' * 70}")
            train.main(run_args)

    compare.main()


if __name__ == "__main__":
    main()
