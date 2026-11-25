"""Single entry point: runs every pipeline stage in order.

    python run_all.py          reproduce all results from saved settings
    python run_all.py --tune   also rerun the hyperparameter search (slow)

Stages that are not built yet are skipped with a message.
"""

import argparse

from src import s1_ingest
from src.utils import log, stage

# Update this after a full timed run (course requirement: show an estimate).
ESTIMATED_MINUTES = 5


def main() -> None:
    """Parse arguments and run the stages in order."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="rerun the hyperparameter search")
    args = parser.parse_args()

    log(f"Crash severity pipeline. Estimated total runtime: about {ESTIMATED_MINUTES} minutes.")
    with stage("Stage 1: load crash data"):
        s1_ingest.run()

    # Stages 2-5 (weather join, features, training, evaluation) plug in here.
    log("Stages 2-5 are not built yet.")
    log("Pipeline completed.")


if __name__ == "__main__":
    main()
