# Hillsborough County Crash Severity Prediction

A course data mining project that predicts traffic crash severity (no injury / injury / severe) in Hillsborough County, FL, from FDOT crash records and NOAA hourly weather, 2014–2018.

**Start here: [KNOWLEDGE.md](KNOWLEDGE.md).** It's the team's source of truth for decisions, deadlines, owners, data fields and rules.

## Run

> The pipeline is not built yet. These are the target instructions.

```bash
pip install -r requirements.txt
python run_all.py          # reproduce all results from saved hyperparameters
python run_all.py --tune   # rerun the hyperparameter search (slow)
```

Figures and tables are written to `outputs/`.

## Libraries

pandas, numpy, scikit-learn, pyarrow, requests, matplotlib (versions pinned in `requirements.txt`).

## Team

Katya Serechenko, Alexander Hsieh, Jaliah Meade, Sid Nguyen

## Citations

_List any borrowed code here, with a link, and add a comment at the snippet itself._
