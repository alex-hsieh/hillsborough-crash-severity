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
| D6 | **Revised Nov 24 after checking the real data.** `CRASH_DATE` is stored as midnight in fixed EST (always 05:00 UTC, all year), so shift it by −5 h to get the calendar date. `CRASH_TIME` ("HHMM") is the officer's local wall-clock time, which **does** follow daylight saving. Date + time = local time, converted to UTC with `America/New_York`. Repeated 1 am in November = daylight occurrence; skipped 2 am in March = shifted forward | The Oct 9 "fixed EST" finding is true of the date field only. Applying a fixed −5 h to the clock time would shift every summer crash by an hour. To be confirmed with the weather join (rain reports should line up with NOAA rain hours) | Nov 24 (replaces the Oct 9 version) |
| D7 | Stretch goals (weather vs. traffic volume; unusual-volume detection) only if core results freeze by Nov 29 | Not enough time otherwise | Nov 16 |
| D8 | Libraries limited to pandas, numpy, scikit-learn, pyarrow, requests, matplotlib | Every extra library is one more thing the instructor must install | Nov 16 |
| D9 | **Crash data is downloaded once from a browser** with `tools/fdot_browser_download.js`, then read from `data/raw/` | gis.fdot.gov now blocks scripts (Python, curl) with a bot check. The county's public copy lacks crash times and speed limits | Nov 24 |
| D10 | **The data ships with the submission** (processed tables in the zip, or a download link if it's too big) | The instructor's machine can't download from FDOT either, so stage 1 can't fetch crash data automatically | Nov 24 |
| D11 | Python 3.14 with the versions pinned in `requirements.txt` | Tested setup; everyone uses the same versions | Nov 24 |

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
| Label | `INJSEVER` (a 1-character **string**: "1" to "6") |
| Time and place | `CRASH_DATE` (date), `CRASH_TIME` ("HHMM" string), `CALENDAR_YEAR` (integer), `COUNTY_TXT`. Coordinates: `OFFICER_LATITUDE`/`OFFICER_LONGITUDE` (officer), `SAFETYLAT`/`SAFETYLON` (FDOT-verified), `LATITUDE`/`LONGITUDE` (basemap-snapped). The layer's native projection is UTM 17N (EPSG 26917); the download script also saves WGS84 `lon_wgs84`/`lat_wgs84` |
| Roadway ID | `ROADWAYID` (FDOT roadway), `ON_ROADWAY_NAME`, `INT_ROADWAY_NAME` |
| Extra roadway fields (added Nov 24) | `RDWY_ALIGN_CD` (alignment), `RDWY_GRDE_CD` (grade), `TRAF_WAY_CD` (traffic way), `JCT_CD` (relation to junction), `V1TRAFCTL` (traffic control), `RCI_MEDIAN_WIDTH_FT`, `RCI_SURFACE_WIDTH_FT` |
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

**Never use these as features (leakage deny list):** `NUMBER_OF_INJURED`, `NUMBER_OF_KILLED`, `NUMBER_OF_SERIOUS_INJURIES`, `TOTAL_PERSONS`, and FDOT's pre-built behaviour flags (`NO_BELT_*`, `IMPAIRED_*`, `SPEEDING_*`, `AGGRESSIVE_*`, `DISTRACTED_*`, `ALCINVCD`, `WRONGWAY_IND`, `LANE_DEPARTURE_IND`, `CRASH_CLASS_IND_MC`). `config.is_denied()` enforces this, and the training code asserts that no feature is denied. `INTERSECTION_IND` and `GEO_URBAN_RURAL_IND` describe the location, so they're allowed.

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
| Nov 24 | Alex | Downloaded data: 2014 24,621 / 2015 27,914 / 2016 30,457 / 2017 29,758 crashes. 2017 exactly matches the Oct 9 check (56.6% / 39.2% / 4.2%) | Download and label mapping confirmed correct |
| Nov 24 | Alex | 197 crashes appear twice with the same crash ID (`XID`) but different OBJECTIDs | Keep the first copy per `XID` |
| Nov 24 | Alex | 1,596 crashes (1.4%) have `INJSEVER` 0 (unknown) or 6 (non-traffic) | Dropped from modeling, as planned |
| Nov 24 | Alex | Speed limit, functional class, lanes, median and surface width, urban/rural and ADT are 46–54% missing. They come from FDOT's roadway inventory, which covers state-maintained roads only: ADT is 1–6% missing on interstate/US/state roads, but 61–88% missing on county and local roads | Keep them with gradient boosting's native missing handling; add road system (`DHSRDSYS`, always filled) as a feature; state this limitation in the report |
| Nov 24 | Alex | Severe share drifts down: 5.8% (2014), 4.6%, 4.2%, 4.2% (2017). 2014 also has ~20% fewer crashes | Possibly the 2014 coding shift; the time-ordered split handles it; mention in the report |
| Nov 24 | Alex | gis.fdot.gov returns an Azure bot-check page to scripts. ArcGIS Hub has no export of `Crashes_All`. Hillsborough County's public copy (Signal Four, 244k crashes 2011–2023) has dates without times and no speed limit, lane count or traffic volume | Download from a browser tab with `tools/fdot_browser_download.js` (D9); ship the data with the code (D10) |

---

## 12. Open questions

- [x] Exact coordinate field names in the crash layer (section 5)
- [ ] Missing rate of `AVERAGE_DAILY_TRAFFIC` in Hillsborough (Alex)
- [ ] Tampa International GHCNh station ID, column names, UTC check, 2014–2018 coverage (Alex)
- [ ] Verify fixed-EST crash times on a few known crashes, e.g. Hurricane Irma, Sept 10–11, 2017 should show heavy rain and wind (Alex, Jaliah double-checks)
- [x] Python and library versions: Python 3.14, pinned in `requirements.txt` (D11)
- [ ] Download the crash data with `tools/fdot_browser_download.js` and put the files in `data/raw/` (Alex)
- [ ] **Canvas zip size limit.** Now important: the data must ship in the zip or by link (D10) (Sid)
- [ ] Everyone confirms their workstream and Thanksgiving availability

---

## 13. Background documents

- [`docs/implementation-plan.md`](docs/implementation-plan.md): the full Nov 16 plan. **Its time zone step and hourly traffic plan are superseded** by D4 and D6 above.
- [`docs/status-handoff-2026-11-24.md`](docs/status-handoff-2026-11-24.md): the Nov 24 status snapshot
