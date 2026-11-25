"""Single entry point: runs every pipeline stage in order.

    python run_all.py          reproduce all results from saved settings
    python run_all.py --tune   also rerun the hyperparameter search (slow)

Before the first run, the FDOT crash files must be in data/raw/
(see tools/fdot_browser_download.js). Weather downloads automatically.
Results are written to outputs/tables/ and outputs/figures/.
"""

import argparse

from src import s1_ingest, s2_integrate, s3_features, s4_train, s5_evaluate
from src.utils import log, stage

# Measured on a 12-core laptop, Nov 25 2026: 51 s default, about 6 min with --tune.
ESTIMATED_MINUTES = 1
ESTIMATED_MINUTES_TUNE = 7


def main() -> None:
    """Parse arguments and run the five stages in order."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="rerun the hyperparameter search")
    args = parser.parse_args()

    minutes = ESTIMATED_MINUTES_TUNE if args.tune else ESTIMATED_MINUTES
    log(f"Crash severity pipeline. Estimated total runtime: about {minutes} minutes.")
    with stage("Stage 1: load crash data"):
        s1_ingest.run()
    with stage("Stage 2: join hourly weather"):
        s2_integrate.run()
    with stage("Stage 3: build label and features"):
        s3_features.run()
    with stage("Stage 4: train models"):
        s4_train.run(tune_params=args.tune)
    with stage("Stage 5: evaluate on 2018"):
        s5_evaluate.run()
    log("Pipeline completed. Tables are in outputs/tables/, figures in outputs/figures/.")


if __name__ == "__main__":
    main()
