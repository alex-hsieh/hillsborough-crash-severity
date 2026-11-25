# Hillsborough County Crash Severity Prediction

A course data mining project that predicts traffic crash severity (no injury / injury / severe) in Hillsborough County, FL, from FDOT crash records and NOAA hourly weather, 2014–2018.

**Start here: [KNOWLEDGE.md](KNOWLEDGE.md).** It's the team's source of truth for decisions, deadlines, owners, data fields and rules.

## Setup

Requires Python 3.14 (other 3.12+ versions will likely work).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- **Crash records (FDOT).** `gis.fdot.gov` blocks scripted downloads, so these files are fetched once from a browser with [`tools/fdot_browser_download.js`](tools/fdot_browser_download.js) (instructions at the top of that file) and placed in `data/raw/`:
  `fdot_crashes_layer_metadata.json` and `fdot_crashes_hillsborough_2014.json` … `_2018.json`.
- **Hourly weather (NOAA GHCNh, Tampa International Airport).** Downloaded automatically by stage 2 if missing.

## Run

```bash
python run_all.py          # reproduce all results from saved settings (~1 min)
python run_all.py --tune   # also rerun the hyperparameter search (~7 min)
```

Each stage prints its progress and elapsed time. Results are written to:

- `outputs/tables/`: data quality, class shares, validation and test metrics, ablation, importance
- `outputs/figures/`: confusion matrices, permutation importance, ablation chart

| Stage | File | What it does |
| --- | --- | --- |
| 1 | `src/s1_ingest.py` | Load FDOT crash files, remove duplicates, build DST-aware timestamps |
| 2 | `src/s2_integrate.py` | Join hourly airport weather to each crash; time zone and Hurricane Irma checks |
| 3 | `src/s3_features.py` | 3-class label, features, leakage checks |
| 4 | `src/s4_train.py` | Five models, tuned on time-ordered folds (2015, 2016, 2017), refit on 2014–2017 |
| 5 | `src/s5_evaluate.py` | Score once on 2018, permutation importance, feature-group ablation, figures |

Settings (years, seeds, label map, feature groups, leakage deny list) live in [`config.py`](config.py).

## Libraries

pandas, numpy, scikit-learn, pyarrow, requests, matplotlib (versions pinned in `requirements.txt`). `joblib` comes with scikit-learn.

## Team

Katya Serechenko, Alexander Hsieh, Jaliah Meade, Sid Nguyen

## Citations

- FDOT State Safety Office, *Crashes_All* feature service: https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer
- NOAA NCEI, *Global Historical Climatology Network hourly (GHCNh)*: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
- scikit-learn: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011

_List any borrowed code here, with a link, and add a comment at the snippet itself._
