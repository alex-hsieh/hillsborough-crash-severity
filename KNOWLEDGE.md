# Project Knowledge Base: Hillsborough County Crash Severity

**This file is the team's source of truth.** If something here is wrong or out of date, fix it in a commit and tell the team. Decisions made in chat don't count until they're written here.

Last updated: Nov 24, 2026

---

## 1. The project in one paragraph

We predict how severe a traffic crash is in Hillsborough County, FL, using FDOT crash records from 2014 to 2018 plus hourly NOAA weather. Each crash gets one of three labels: **no injury, injury, or severe** (incapacitating or fatal). The main model is gradient boosting (`HistGradientBoostingClassifier`), and we compare it with four simpler models. We train on 2014–2017, test once on 2018, and report **macro-F1**. Then we measure which features matter (permutation importance) and whether weather and traffic actually help (feature-group ablation).

**Research question (from the proposal):** "Can Hillsborough County traffic crash severity be predicted using factors such as traffic conditions, roadway characteristics, crash location, and weather?"

---

## 2. Deadlines

| Date | Milestone |
| --- | --- |
| Wed Nov 25 | Crash data downloaded and cleaned; crash-only model runs end to end |
| Nov 26–27 | Thanksgiving: light work only |
| Sat Nov 28 | Weather joined to crashes |
| **Sun Nov 29** | **Results freeze: no model changes after this** |
| Tue Dec 1 | Report draft complete |
| Wed Dec 2 | Clean-machine test run passes, zip ready (internal finish line) |
| Thu Dec 3 | Buffer |
| **Fri Dec 4, 11:59 pm** | **Submit on Canvas** |

---

## 3. Team and ownership

Alex builds the full code pipeline. Everyone else focuses on the report, checking results, and packaging. Each person replies to confirm or swap, then updates this table.

| Person | Workstream | Delivers | Due |
| --- | --- | --- | --- |
| Alexander Hsieh (Alex) | Full pipeline | Crash download and cleaning, weather join, label and features, all models, tuning, evaluation, figures and tables, `run_all.py`, `config.py` | Nov 29 |
| Katya Serechenko | Report sections 1–2 | Introduction and applications, literature survey (~1.75 pages), starting from the proposal | Dec 1 |
| Jaliah Meade | Report section 3 + results check | Methodology (~1.5 pages); review data and results tables as they land in the repo and flag anything that looks off | Dec 1 |
| Sid Nguyen | Packaging | README, test run on a clean machine, final zip | Dec 2 |
| Everyone | Report sections 5–6 | Once results freeze (Nov 29): results discussion and conclusion | Dec 1 |

Alex writes section 4 (implementation) from the log in section 11. **Anyone** who hits a problem or makes an assumption adds it to that log.

**Thanksgiving availability:** _each person fill in which days you're reachable._

---

## 4. Decisions (final unless changed here)

| # | Decision | Why | Date |
| --- | --- | --- | --- |
| D1 | Hillsborough County only, 2014–2018 | Counties differ too much to pool (e.g. 19% injury crashes in Miami-Dade, 39% in Hillsborough); FDOT coding shifted around 2014 | Oct 9 |
| D2 | 3-class label, not binary | Fatal crashes are too rare to model alone, so they're merged with incapacitating | Oct 9 |
| D3 | Train 2014–2017, test 2018 exactly once | Time-based split, so no future data leaks into training | Oct 9 |
| D4 | **Traffic = `AVERAGE_DAILY_TRAFFIC` field from the crash record.** No hourly traffic counts | The hourly bulk download was never confirmed; this field needs no download or join | Nov 24 |
| D5 | **Weather = Tampa International Airport station only.** Add more stations only if time allows | One station keeps the join simple; still needs a check for 2014–2018 coverage | Nov 24 |
| D6 | **Crash times are fixed EST with NO daylight saving.** Do NOT use `America/New_York` | FDOT layer metadata says so. Applying DST shifts summer crashes by an hour against the weather | Oct 9 (overrides the Nov 16 plan) |
| D7 | Stretch goals (weather vs. traffic volume; unusual-volume detection) only if core results freeze by Nov 29 | Not enough time otherwise | Nov 16 |
| D8 | Libraries limited to pandas, numpy, scikit-learn, pyarrow, requests, matplotlib | Every extra library is one more thing the instructor must install | Nov 16 |

---

## 5. Data sources

### Crash records (FDOT)

- **Service:** [Crashes_All FeatureServer](https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer), layer 0 "All Crashes"
- **Filter:** `COUNTY_TXT` = Hillsborough, `CALENDAR_YEAR` 2014–2018
- **Limits:** JSON only, **max 1,000 records per request**. About 30,000 crashes a year means about 150 paged requests. Add retries.
- **Coverage:** 2011 to early 2019 (2012–2018 are complete years)

**Verified field names:**

| Role | Fields |
| --- | --- |
| Label | `INJSEVER` |
| Time and place | `CRASH_DATE`, `CRASH_TIME`, `CALENDAR_YEAR`, `COUNTY_TXT`, plus three coordinate sets (officer-reported, FDOT-verified, basemap-snapped). _Exact coordinate field names: TODO, Alex to fill in_ |
| Conditions | `LGHT_COND_CD` (lighting), `EVNT_WTHR_COND_CD` (officer-reported weather), `RD_SRFC_COND_CD` (road surface) |
| Roadway | `SPEED_LIMIT`, `FUNCLASS`, `CNTOFLANES`, `GEO_URBAN_RURAL_IND`, `INTERSECTION_IND`, `IMPCT_TYP_CD` (manner of collision) |
| Traffic | `AVERAGE_DAILY_TRAFFIC` (probably empty off the state highway system; check the missing rate) |

### Hourly weather (NOAA GHCNh)

- **Product page:** [GHCNh](https://ncei.noaa.gov/products/global-historical-climatology-network-hourly)
- **Bulk access:** AWS `s3://noaa-ghcnh-pds` (no credentials), PSV or parquet, one file per station per year
- **Station list:** [ghcnh-station-list.txt](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt)
- **Docs:** [ghcnh_DOCUMENTATION.pdf](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf)
- **Known:** visibility is in km
- **TODO (Alex):** confirm the Tampa International station ID, the exact column names, that timestamps are UTC, and coverage for 2014–2018

### Traffic count sites (FDOT), not used, see D4

- [Traffic_TMSCOUNT_TDA FeatureServer](https://services1.arcgis.com/O1JpcwDW8sjYuddV/ArcGIS/rest/services/Traffic_TMSCOUNT_TDA/FeatureServer): 23 sites in Hillsborough (current snapshot). Kept for reference only.

---

## 6. Label definition

Built from `INJSEVER` ("Highest Injury in Crash").

| Class | Code | `INJSEVER` values |
| --- | --- | --- |
| No injury | 0 | 1 |
| Injury | 1 | 2, 3 |
| Severe | 2 | 4 (incapacitating), 5 (fatal) |
| **Dropped** | — | 0 (unknown / not coded), 6 (non-traffic) |

**Sanity check, Hillsborough 2017:** 29,758 crashes, about 56% no injury, 39% injury, 4.2% severe (1,243). If your label doesn't roughly match this, something is wrong.

---

## 7. Features and leakage rules

**Feature groups** (these are also the ablation steps, defined once in `config.py`):

| Group | Features |
| --- | --- |
| 1. Roadway and crash | `SPEED_LIMIT`, `FUNCLASS`, `CNTOFLANES`, `GEO_URBAN_RURAL_IND`, `INTERSECTION_IND`, `IMPCT_TYP_CD`, latitude, longitude |
| 2. Time | hour, day of week, month, weekend flag, `LGHT_COND_CD` |
| 3. Weather | NOAA temperature, precipitation, visibility, wind speed, humidity, missing flags; officer-reported `EVNT_WTHR_COND_CD`, `RD_SRFC_COND_CD` _(TODO: confirm they belong in this group)_ |
| 4. Traffic | `AVERAGE_DAILY_TRAFFIC` + missing flag |

**Never use these as features (leakage deny list):** `NUMBER_OF_INJURED`, `NUMBER_OF_KILLED`, `NUMBER_OF_SERIOUS_INJURIES`, `TOTAL_PERSONS`, and every pre-built indicator flag. The training code asserts that none of them is in the feature list.

**Other rules:**
- Anything learned from data (imputation values, category lists) is fit on training years only.
- If one feature carries almost all the importance, or macro-F1 is far above every other model, assume a leaked column and check.

---

## 8. Models and evaluation

| # | Model | Purpose |
| --- | --- | --- |
| 1 | `DummyClassifier` (majority class) | The floor: accuracy about 56%, macro-F1 about 0.24 |
| 2 | `LogisticRegression` (multinomial, balanced) | Linear baseline |
| 3 | `DecisionTreeClassifier` (depth 4–12) | Interpretable reference |
| 4 | `RandomForestClassifier` (~300 trees, balanced) | Strong baseline |
| 5 | `HistGradientBoostingClassifier` | **Primary model**: ~30-trial random search, extra weight on the severe class |

- **Tuning:** three time-ordered folds (train ≤2014 → validate 2015; ≤2015 → 2016; ≤2016 → 2017)
- **Final:** refit on 2014–2017, score 2018 once
- **Metrics:** macro-F1 (headline), per-class precision and recall, accuracy (context only), confusion matrix, ordinal MAE
- **Interpretation:** permutation importance (top 15), feature-group ablation (group 1 → +2 → +3 → +4), 3-seed noise check
- A result where weather and traffic add little is **still a valid result**. Report it honestly; don't tune until it changes.

---

## 9. Code layout and run contract

```
run_all.py           # ONE command runs everything: python run_all.py
config.py            # paths, years, seeds, feature groups, deny list
requirements.txt     # pinned versions
src/
  s1_ingest.py       # download or load cached raw data
  s2_integrate.py    # join weather to crashes
  s3_features.py     # label, features, year splits
  s4_train.py        # all five models
  s5_evaluate.py     # metrics, importance, ablation, figures
  utils.py           # logging, timers, helpers
data/                # NOT committed (see .gitignore)
  raw/  interim/  processed/
outputs/
  figures/  tables/  models/
```

**Rules:**
- `python run_all.py` reproduces the results from saved hyperparameters (`outputs/models/best_params.json`). `--tune` reruns the search.
- Stage 1 downloads data only if it isn't already in `data/raw/`.
- Every stage prints its name, row counts, and elapsed time. The script prints the estimated total runtime at the start.
- All random seeds are set in `config.py`.
- Every function has a docstring. Borrowed code gets a comment with its source and a line in the README citations.
- Target runtime: under about 15 minutes.

**Git workflow:**
- Until the Nov 29 freeze, Alex pushes pipeline code straight to `main` to keep moving. Jaliah reviews the results as they land.
- Everyone else works on a branch and opens a PR to `main`.
- Never commit raw data or large files. Each stage writes its own checkpoint.

---

## 10. Course requirements (Project Details sheet)

- Report: **at most 8 pages** excluding references, single spaced, font 10–12, PDF
- Six required sections:

| Section | Pages | Owner |
| --- | --- | --- |
| 1. Introduction and applications | 0.75 | Katya |
| 2. Literature survey | 1 | Katya |
| 3. Methodology | 1.5 | Jaliah |
| 4. Implementation (data access, joins, problems, assumptions) | 1.25 | Alex, from the log below |
| 5. Experimental results | 2.75 | Alex (tables and figures) + everyone (discussion) |
| 6. Conclusion and future work | 0.5 | Everyone, written last |

- Code must run end to end from one command, with progress messages and an estimated completion time
- README with run instructions and the libraries used
- The instructor must be able to reproduce the report's results
- Borrowed code must be cited; ask the instructor when unsure
- **One zip per team** (report PDF + all code). One person submits, a second confirms the upload opens.
- Worth **40% of the grade**. A presentation may be required; details will come on Canvas.
- Sample reports 1–4 are in the course files. Worth a skim before writing.

---

## 11. Problems and assumptions log

**Add an entry whenever you hit a problem or make an assumption.** Report section 4 is written straight from this log.

| Date | Who | What happened / what we assumed | Resolution |
| --- | --- | --- | --- |
| Oct 9 | Team | Crash date fields use fixed EST, no DST | Convert with a fixed UTC−5 offset, not `America/New_York` (D6). _TODO: verify on a few known crash times_ |
| Oct 9 | Team | Fatal crashes too rare to model alone | Merged with incapacitating into "severe" (D2) |
| Nov 24 | Alex | Hourly traffic bulk download never confirmed | Use `AVERAGE_DAILY_TRAFFIC` from the crash record (D4) |

---

## 12. Open questions

- [ ] Exact coordinate field names in the crash layer (Alex)
- [ ] Missing rate of `AVERAGE_DAILY_TRAFFIC` in Hillsborough (Alex)
- [ ] Tampa International GHCNh station ID, column names, UTC check, 2014–2018 coverage (Alex)
- [ ] Verify fixed-EST crash times on a few known crashes, e.g. Hurricane Irma, Sept 10–11, 2017 should show heavy rain and wind (Alex, Jaliah double-checks)
- [ ] Python and scikit-learn versions everyone uses (Alex picks and pins them in `requirements.txt`)
- [ ] Canvas zip size limit, which decides whether processed data ships in the zip (Sid)
- [ ] Everyone confirms their workstream and Thanksgiving availability

---

## 13. Background documents

- [`docs/implementation-plan.md`](docs/implementation-plan.md): the full Nov 16 plan. **Its time zone step and hourly traffic plan are superseded** by D4 and D6 above.
- [`docs/status-handoff-2026-11-24.md`](docs/status-handoff-2026-11-24.md): the Nov 24 status snapshot
