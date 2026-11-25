# Implementation Plan: Hillsborough County Crash Severity Prediction

Nov 16, 2026 · @Alexander

## Where we stand

Eighteen days remain before the final report and code are due on Friday, Dec 4 at 11:59 pm, and Thanksgiving week sits in the middle. This plan treats Wednesday, Dec 2 as the internal finish line so the last two days are buffer.

The proposal's timeline had data downloaded by late October and baselines running by early November. This plan starts from data acquisition, so skip any phase the team has already finished and pull the later dates forward.

| Scope | What it covers | Rule |
| --- | --- | --- |
| Core (must ship) | 3-class severity model with gradient boosting, four comparison models, 2018 test evaluation, permutation importance, feature-group ablation | Everything in the schedule below serves this |
| Stretch (proposal said "if time permits") | Weather versus traffic volume and crash frequency; unusual traffic volume detection | Start only if core results are frozen by Sunday, Nov 29 |
| Cut unless free | Extra model families, SHAP, hyperparameter searches beyond 30 trials | Mention as future work in the report |

One decision to make today: confirm with the team which phases already have working code, and who owns each remaining phase (a suggested split is near the end).

## Deliverables and course requirements

The submission is one zip on Canvas holding the report PDF and all Python code, and the code has to run end to end from a single command on the instructor's machine. Every row below comes from the course's Project Details sheet.

| Requirement | What it means for this project |
| --- | --- |
| Report of at most 8 pages, single spaced, font 10 to 12, references excluded, six required sections | Page budget per section is at the end of this plan |
| Code runs in sequence from one trigger until visual and metric results appear | `python run_all.py` runs all five stages and writes every figure and table |
| Estimated completion time and real-time progress messages | Print the estimate at start; each stage logs start, finish, and elapsed time ("Training completed") |
| Adequate comments | A docstring on every module and function, plus comments on any non-obvious join or filter |
| README with run instructions and the libraries used | `README.md` plus a pinned `requirements.txt` |
| Instructor can run the code and reproduce the report's results | Fixed random seeds, a data snapshot that ships with the code or downloads automatically, and a test run on a clean machine |
| Borrowed code is cited | A citations list in the README and a comment at each borrowed snippet |
| Possible in-class presentation, details on Canvas later | Reuse the report figures; no separate work planned yet |
| One submission per team | One person submits; a second person confirms the upload opened correctly |

## Repository layout and run contract

One script, `run_all.py`, calls five stage modules in order, and each stage writes a checkpoint file so a teammate can rerun one stage without redoing the others.

&#91;embedded content: pipeline · 3 sources, 5 stages\]

Each stage reads the previous stage's checkpoint, so the three sources only meet in stage 2 and the model stages never touch raw files.

```
crash-severity/
  run_all.py           # single trigger: runs stages 1 to 5 in order
  config.py            # paths, years, seeds, station list, feature groups
  requirements.txt     # pinned versions
  README.md
  src/
    s1_ingest.py       # download or load cached raw data, basic cleaning
    s2_integrate.py    # join weather and traffic volume to crashes
    s3_features.py     # severity label, features, year-based splits
    s4_train.py        # four comparison models + gradient boosting
    s5_evaluate.py     # metrics, importance, ablation, figures
    utils.py           # logging, timers, distance helpers
  data/
    raw/               # as downloaded, never edited by hand
    interim/           # cleaned table per source (parquet)
    processed/         # one model-ready table (parquet)
  outputs/
    figures/  tables/  models/
```

Rules the code follows so the instructor's run matches the report:

- **Default mode reproduces, it does not search.** `python run_all.py` loads the saved best hyperparameters from `outputs/models/best_params.json` and refits. `python run_all.py --tune` reruns the full search.
- **Data is cached.** Stage 1 uses files in `data/raw/` when they exist and downloads only when they are missing.
- **Every stage reports progress.** Print the stage name, row counts in and out, and elapsed time. Print the total estimated runtime at the start, measured from a real run.
- **Seeds are fixed in one place.** `config.py` holds the seed, the year ranges, and the feature-group lists used by the ablation.
- **Libraries:** pandas, numpy, scikit-learn, pyarrow, requests, matplotlib. Add nothing else without telling the team, since each addition is one more thing the instructor installs.

## Phase 1: Data acquisition

Three raw datasets land in `data/raw/` by Wednesday, Nov 18, and the hourly traffic counts are the one source whose availability is still unconfirmed.

| Source | Where to get it | What to pull | Watch for |
| --- | --- | --- | --- |
| Crash records | [FDOT Crashes\_All feature service](https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer), layer 0 "All Crashes", or the same data exported from the FDOT Open Data Hub | Hillsborough County, 2014 through 2018, all attribute fields plus coordinates | The service returns JSON only and at most 1,000 records per request. At roughly 30,000 crashes a year, plan for about 150 paged requests with retries |
| Hourly weather | [NOAA GHCNh](https://ncei.noaa.gov/products/global-historical-climatology-network-hourly) bulk download, one file per station per year, PSV or parquet | 2014 through 2018 for every station in or near the county. Pick them from the [station list](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt) by latitude and longitude | Confirm the exact column names and that times are in UTC from the [documentation PDF](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf) on first load |
| Hourly traffic volume | [FDOT traffic monitoring site layer](https://services1.arcgis.com/O1JpcwDW8sjYuddV/ArcGIS/rest/services/Traffic_TMSCOUNT_TDA/FeatureServer) for site locations; hourly counts through Florida Traffic Online | Site ID, coordinates, roadway ID, and hourly counts by direction for the county's continuous sites, 2014 through 2018 | The layer says the sites have hourly volume by direction, but it does not show whether 2014 to 2018 hourly counts can be downloaded in bulk. Settle this first |

Tasks:

- [ ] Write the paged crash download with a county and year filter, and save the result as one parquet file
- [ ] Select weather stations within about 40 km of the county and download their yearly files
- [ ] Try to export hourly counts for one continuous site for one month of 2017; if that works, script the rest
- [ ] If hourly counts are not downloadable, email FDOT Transportation Data and Analytics the same day and switch to the fallback in the risks section
- [ ] Write a data dictionary: fields kept, row counts, and percent missing per field for each source

Done when: all three sources load into pandas from `data/raw/`, and the missing-value table exists. That table goes straight into the report.

## Phase 2: Cleaning and integration

The output is one table with one row per crash, carrying the matched weather and traffic volume plus a distance and a missing flag for each match. No crash row is dropped because a match failed.

1. **Clean crashes.** Remove duplicate report numbers, drop rows with missing or out-of-county coordinates, and build one timezone-aware timestamp from the crash date and time. Crash times are local, so localize to `America/New_York` and convert to UTC. Decide once how to treat the repeated hour at the end of daylight saving time and write the choice down.
2. **Clean weather.** Stations report more than once an hour. Reduce each station to one row per UTC hour, keeping the observation closest to the top of the hour, and keep the fields the model needs: temperature, precipitation, visibility, wind speed, relative humidity.
3. **Join weather.** Find each crash's nearest station by great-circle distance (scikit-learn's `BallTree` with the haversine metric handles all crashes in one call). Round the crash time to the nearest hour and merge on station and hour. If that station has no row for the hour, fall back to the next nearest station. Store `wx_station_km` and a missing flag per weather field.
4. **Join traffic volume.** Match each crash to the nearest continuous count site on the same roadway ID, then merge on site, date, and hour. Store `count_site_km` and `volume_missing`. Most local-road crashes will have no site on their road; leave their volume empty.
5. **Verify every join.** Assert the row count is unchanged after each merge. Print the match rate and the distance distribution for both joins. Spot-check a known event: crashes during Hurricane Irma (Sept 10 to 11, 2017) should show heavy rain and high wind.

Done when: `data/interim/crashes_joined.parquet` exists, the row count equals the cleaned crash count, and the match rates are written to `outputs/tables/`.

The time zone conversion is the step most likely to be silently wrong. An off-by-one-hour or off-by-five-hour error still produces a full table, and only the spot check catches it.

## Phase 3: Labels, features, and leakage controls

The label has three classes built from the FDOT injury severity field, and every feature must be something known at the scene. Build this phase on crash-only columns first so modeling can start before the weather and traffic joins finish.

**Label.** Map the severity field as the proposal defines it, then check the result against the proposal's 2017 figures (about 56% no injury, 39% injury, 4% severe, of roughly 29,800 crashes).

| Class | Coded as | FDOT severity values |
| --- | --- | --- |
| No injury | 0 | No injury |
| Injury | 1 | Possible injury, non-incapacitating injury |
| Severe | 2 | Incapacitating injury, fatal |
| Dropped |  | Unknown severity, non-traffic fatality |

**Features by group.** These four groups are also the ablation steps, so define them once in `config.py`.

| Group | Features | Notes |
| --- | --- | --- |
| Roadway and crash | Road type, speed limit, intersection presence, collision type, latitude, longitude | Add lane count or median type only if the crash layer already has them |
| Time | Hour of day, day of week, month, weekend flag, lighting condition | Keep hour as a number; the trees find the splits |
| Weather | Temperature, precipitation, visibility, wind speed, humidity, station distance, missing flags | Add rain in the previous 3 hours as a wet-road proxy if time allows |
| Traffic | Hourly volume at the matched site, volume relative to that site's usual level for the hour and weekday, site distance, missing flag | Compute the "usual level" from training years only |

**Leakage controls.**

- Keep an explicit deny list in `config.py`: injury counts, fatality counts, total persons, and every pre-built indicator flag. Assert at the start of training that no denied column is in the feature list.
- Fit anything learned from data (typical volumes, imputation values, category lists) on training years only.
- After the first model trains, look at the top five features. One feature carrying nearly all the importance, or a macro-F1 far above the comparison models, means a leaked column.

**Encoding.** Store categorical fields with pandas `category` dtype for gradient boosting, which reads them directly and accepts up to 255 categories per feature. Logistic regression gets one-hot encoding and scaling inside a scikit-learn `Pipeline`; the tree and forest get ordinal codes plus median imputation with a missing indicator.

**Splits.** Tune on 2014 to 2017 with three time-ordered folds (train through 2014, validate 2015; through 2015, validate 2016; through 2016, validate 2017). Refit the final models on all of 2014 to 2017 and score 2018 exactly once.

Done when: `data/processed/model_table.parquet` has the label, a `year` column, and only allowed features, and the class shares by year are saved as a table.

## Phase 4: Model training

Five models train on the same features and the same year splits, built in order of complexity so there is always a working result to report.

| Order | Model | scikit-learn class | Setup |
| --- | --- | --- | --- |
| 1 | Majority class | `DummyClassifier` | The floor. If 2018 resembles 2017, expect accuracy near 56% and macro-F1 near 0.24 |
| 2 | Multinomial logistic regression | `LogisticRegression` | One-hot and scaled inputs, balanced class weights |
| 3 | Decision tree | `DecisionTreeClassifier` | Depth tuned between 4 and 12; plot the top three levels as the interpretable reference |
| 4 | Random forest | `RandomForestClassifier` | About 300 trees, minimum leaf size tuned, balanced class weights |
| 5 | Gradient boosting (primary) | `HistGradientBoostingClassifier` | Native categorical and missing-value handling, sample weights, randomized search |

**Gradient boosting search.** Run about 30 random configurations across the three time-ordered folds, scored by macro-F1.

| Hyperparameter | Range |
| --- | --- |
| `learning_rate` | 0.03 to 0.3, sampled on a log scale |
| `max_iter` | Up to 1,000, cut short by early stopping |
| `max_leaf_nodes` | 15 to 127 |
| `max_depth` | None, 4, 6, 8, 10 |
| `min_samples_leaf` | 20 to 200 |
| `l2_regularization` | 0 to 10 |

Two details that are easy to get wrong:

- **Early stopping and time order.** The classifier's built-in early stopping holds out a random slice of the training data, which mixes years. Prefer choosing the iteration count on the fold's validation year, either by passing an explicit validation set if the installed scikit-learn version supports it, or with `staged_predict`.
- **Class weights.** Start from balanced sample weights. Then try a multiplier on the severe class (1, 2, 4, 8) and pick it on the 2017 validation year, reading macro-F1 and severe-class recall together. Record the chosen value and the trade-off it made, since the report needs both.

Save the winning configuration to `outputs/models/best_params.json` so the default run refits it without searching.

Done when: all five models produce 2017 validation scores from one call to `s4_train.py`, and gradient boosting beats the majority-class floor on macro-F1 by a clear margin.

## Phase 5: Evaluation and interpretation

Macro-F1 on 2018 is the headline number, and every figure and table in the report is written to `outputs/` by `s5_evaluate.py` so nothing is produced by hand.

**Metrics, computed for all five models on 2018:**

- Macro-F1 (headline) and per-class precision and recall
- Accuracy, shown for context only
- Confusion matrix, normalized by true class
- Ordinal mean absolute error: code the classes 0, 1, 2 and average the absolute gap between predicted and true class, so calling a severe crash "no injury" costs 2 and calling it "injury" costs 1

**Interpretation:**

- **Permutation importance** on the 2018 test set, scored by macro-F1, 5 to 10 repeats. Run it on a random sample of about 20,000 test rows to keep the runtime down, and report the top 15 features with their spread.
- **Feature-group ablation.** Refit gradient boosting four times with the tuned hyperparameters: roadway and crash only, then adding time, then weather, then traffic. Report macro-F1 and severe-class recall at each step.
- **Noise check.** Refit the final model with three seeds. A gain from adding weather or traffic that is smaller than the seed-to-seed spread should be reported as no measurable gain.

**Outputs for the report:**

| Item | Content | Built in |
| --- | --- | --- |
| Table 1 | Rows, fields, and percent missing per data source | Phase 1 |
| Table 2 | Class shares by year, 2014 to 2018 | Phase 3 |
| Table 3 | Five models by macro-F1, per-class precision and recall, accuracy, ordinal error | Phase 5 |
| Table 4 | Ablation: metric at each feature-group step | Phase 5 |
| Figure 1 | Pipeline diagram | This plan |
| Figure 2 | Confusion matrices, gradient boosting beside the best comparison model | Phase 5 |
| Figure 3 | Permutation importance, top 15 features | Phase 5 |
| Figure 4 | Ablation bars: macro-F1 and severe recall by step | Phase 5 |

Done when: one run of `run_all.py` on a fresh checkout regenerates all four tables and figures 2 to 4 with the same numbers.

## Schedule

The order is end to end first: a crash-only model runs through `run_all.py` by Friday, Nov 20, and weather and traffic are added afterward as feature groups. That way a late or failed join costs one ablation step instead of the whole project.

&#91;embedded content: schedule · 8 work items, 3 milestones\]

Thanksgiving (Nov 26) falls inside the tuning and evaluation rows, so those carry the most schedule risk. The stretch analyses fit between Nov 29 and Dec 1 only if the freeze holds, and Dec 3 is left empty as buffer.

## Suggested ownership

Four workstreams map onto four people, with the modeling stream going to Alex, who already owns the algorithm section. The other three rows are open for the team to assign.

| Workstream | Covers | Owner | Hands off |
| --- | --- | --- | --- |
| Crash data and features | Crash download, cleaning, label, roadway and time features, leakage deny list | To assign | `model_table.parquet` (crash-only version) by Nov 20 |
| Weather and traffic integration | GHCNh and count-site downloads, both joins, match-rate tables | To assign | `crashes_joined.parquet` by Nov 23 |
| Modeling | Comparison models, gradient boosting, tuning, class weights, ablation runs | Alex | `best_params.json` and validation scores by Nov 27 |
| Evaluation, report, and packaging | Metrics and figures, report assembly, README, clean-machine run | To assign | Report draft by Dec 1, zip by Dec 2 |

Everyone writes the report section for their own workstream. One person owns `run_all.py` and `config.py` so the stage interfaces stay consistent; the modeling owner is the natural choice.

## Risks and fallbacks

The hourly traffic counts are the largest risk, because the whole traffic feature group depends on a download that has not been confirmed.

| Risk | How you will know | Fallback |
| --- | --- | --- |
| Hourly counts for 2014 to 2018 cannot be downloaded in bulk | No working export for one site by Nov 18 | Use each site's annual average daily traffic as the traffic feature, state the change in the report, and keep the ablation step |
| Few crashes share a roadway with a continuous count site | Match rate well under half after the traffic join | Keep volume with its missing flag, add the nearest site on any road as an area-level congestion signal, and report the match rate |
| Weather fields are often missing, visibility especially | Missing-value table from Phase 1 | Next-nearest station, then leave empty; gradient boosting handles missing values directly |
| Time zone or daylight saving error in the joins | Hurricane Irma spot check shows dry, calm weather | Fix the localization step before any modeling on weather features |
| Severe class (about 4%) is rarely predicted | Severe recall near zero on 2017 | Raise the severe-class weight; report the precision cost |
| A leaked column inflates the scores | One feature dominates importance, or macro-F1 jumps far above the comparison models | Audit against the deny list and rerun |
| Instructor's machine cannot reach FDOT or NOAA | Not knowable in advance | Ship the processed tables in the zip if Canvas allows the size; otherwise host them and have Stage 1 download from that link |
| Full run takes too long to grade | Timed run over about 15 minutes | Default mode skips the search and refits saved hyperparameters |
| Thanksgiving week (Nov 26) thins out availability | Tasks unassigned by Nov 24 | Agree on who is reachable which days before Nov 24; hold the Nov 29 freeze |

A model that shows weather and traffic add little beyond roadway and time features is still a valid result. The proposal frames that as an open question, so report it with the ablation numbers instead of tuning until it changes.

## Final report outline

The six required sections fit in 8 pages if results get about a third of the space. Sections 1 to 3 can be drafted now from the proposal, before any results exist.

| Required section | Pages | Content | Source |
| --- | --- | --- | --- |
| 1. Introduction and applications | 0.75 | Research question; uses such as dispatch triage and identifying high-risk road segments | Proposal section II, expanded |
| 2. Literature survey | 1 | Statistical versus machine learning severity models; where this project differs (time-based split, 3 classes, added weather and volume) | Proposal section III, plus one or two gradient boosting references |
| 3. Methodology | 1.5 | Pipeline figure, label definition, feature groups, models, splits, metrics | Proposal section IV and this plan |
| 4. Implementation | 1.25 | Data access, join logic, problems met, assumptions and why (time zone handling, station distance, unmatched volumes) | Notes kept during Phases 1 to 3 |
| 5. Experimental results | 2.75 | Tables 1 to 4, figures 2 to 4, and a short discussion under each | `outputs/` |
| 6. Conclusion and future work | 0.5 | What predicted severity, what did not, limits, the stretch questions as next steps | Written last |

Keep a shared running log of problems and assumptions from today. Section 4 asks for exactly that, and it is hard to reconstruct in the last week.

## Sources

- [Crashes\_All feature service, FDOT](https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer): layer list, 1,000-record limit, JSON only
- [Traffic\_TMSCOUNT\_TDA feature service, FDOT](https://services1.arcgis.com/O1JpcwDW8sjYuddV/ArcGIS/rest/services/Traffic_TMSCOUNT_TDA/FeatureServer): traffic monitoring site layer
- [Global Historical Climatology Network hourly, NOAA NCEI](https://ncei.noaa.gov/products/global-historical-climatology-network-hourly): access methods, formats, station list, variables
- Course Project Details sheet and the team's project proposal (in the project files): deadlines, deliverables, and all methodology choices

Not checked from here: the crash layer's field names and year coverage, the GHCNh column names and time convention, and the installed scikit-learn version. Each is flagged where it matters above.
