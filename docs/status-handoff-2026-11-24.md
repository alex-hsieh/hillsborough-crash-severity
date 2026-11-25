# Project Status Handoff: Hillsborough County Crash Severity

Nov 24, 2026 · @Alexander

Almost nothing has been recorded since the Nov 16 implementation plan. The only sources are that plan, the Oct 9 proposal chat, the course Project Details sheet and the proposal file. The Oct 9 chat does hold verified facts the plan left out, and one of them contradicts the plan's time zone handling (see sections 7 and 8).

## 1. Progress

The status of every phase is unknown, because no work after the Nov 16 plan is on record.

| Phase | Status |
| --- | --- |
| 1. Data acquisition | Unknown. On Oct 9 the team ran two test queries against the FDOT services and both worked: crash counts by year and severity, and count sites by county. No full download is recorded |
| 2. Cleaning and integration | Unknown |
| 3. Labels and features | Unknown in code. The label grouping was checked against real counts on Oct 9 (sections 5 and 7) |
| 4. Training | Unknown |
| 5. Evaluation | Unknown |
| Report | Unknown. The proposal was due Oct 9 |

Code and data location: unknown. No repo URL, Drive folder or filenames are on record. The `crash-severity/` layout in the plan is a proposed structure, not something that exists.

## 2. Traffic data

None of the three traffic questions has a recorded answer, but an AADT fallback is easier than the plan assumes.

- **Bulk hourly download for 2014 to 2018:** not confirmed. On Oct 9 we found that the sites carry hourly volume by direction and that a daily traffic volume CSV archive exists. How many years of hourly history it holds was not confirmed.
- **Switch to the AADT fallback:** no decision recorded.
- **Reply from FDOT:** none recorded, and no email to FDOT is recorded either.
- **What the plan missed:** the crash layer already has an `AVERAGE_DAILY_TRAFFIC` field on each crash record, so the fallback needs no separate download. It is probably empty for crashes off the state highway system; that is an inference and has not been checked.

## 3. Team

Only Alex's role is on record; the other three workstreams are unassigned as far as anything shows.

- **Members:** Katya Serechenko, Alexander Hsieh (Alex), Jaliah Meade, Sid Nguyen.
- **Ownership:** Alex wrote the proposal's Algorithms section and owns the algorithm and modeling work. Owners of crash data and features, weather and traffic integration, and evaluation, report and packaging are unknown; the plan left them "to assign."
- **Thanksgiving availability:** unknown.

## 4. Schedule

The plan freezes core results on Nov 29 and submits on Dec 4. These are the dates from the plan's schedule chart, set as a table here so they survive export.

| Work item | Start | End |
| --- | --- | --- |
| Data download, schema check | Nov 16 | Nov 18 |
| Crash-only model, end to end | Nov 17 | Nov 20 |
| Weather join | Nov 18 | Nov 22 |
| Traffic volume join | Nov 19 | Nov 23 |
| Gradient boosting and tuning | Nov 21 | Nov 27 |
| Report draft | Nov 23 | Dec 1 |
| Evaluation and ablation | Nov 26 | Nov 29 |
| Clean-machine run, README | Nov 30 | Dec 2 |

| Milestone | Date |
| --- | --- |
| Decide on hourly traffic data | Nov 18 |
| Core results freeze | Nov 29 |
| Submit on Canvas | Dec 4 |

Other dates in the plan: Dec 2 is the internal finish line, Dec 3 is buffer, and the stretch analyses start only if results freeze by Nov 29.

## 5. Proposal details

The severity label comes from `INJSEVER` ("Highest Injury in Crash"), codes 0 to 6; the proposal text calls it only "the FDOT injury severity field."

**Title:** "Hillsborough County Traffic and Crash Analysis Using Machine Learning."

| Class | `INJSEVER` codes |
| --- | --- |
| No injury | 1 |
| Injury | 2 and 3 |
| Severe | 4 (incapacitating) and 5 (fatal) |
| Dropped | 0 (unknown or not coded) and 6 (non-traffic) |

The meaning of 6 was inferred from its counts. Which of 2 and 3 is "possible injury" and which is "non-incapacitating" is unknown; both go into the Injury class, so it changes nothing.

**Primary research question:** "Can Hillsborough County traffic crash severity be predicted using factors such as traffic conditions, roadway characteristics, crash location, and weather?"

**Stretch goals, "if time permits":**

1. How are weather conditions, including precipitation and visibility, associated with traffic volume and crash frequency/severity?
2. Can we identify unusually high or low traffic volumes and check whether they are associated with adverse weather or increased crash activity?

The Oct 9 drafts suggested Poisson or negative binomial regression for the first, and residual-based flags or Isolation Forest for the second.

**Where the proposal and the plan differ:**

- The proposal timeline had anomaly detection and the weather analysis in mid-November as regular work; the plan makes both stretch goals.
- The proposal timeline had data done by late October and baselines by early November.
- The proposal never mentions `AVERAGE_DAILY_TRAFFIC` from the crash record.
- The plan's time zone handling contradicts what was found on Oct 9 (section 8).

## 6. Course requirements beyond the plan

The Project Details sheet gives no rubric weights and no zip size limit; the rest it states is below.

- **Presentation:** "Students may be required to present their projects in class (time permitting). Details will be posted later in Canvas." No details are recorded.
- **Grading:** the only weight given is that the project is worth 40% of the grade. Rubric weights are unknown.
- **Zip size limit:** not stated in the sheet, so unknown.
- **Report formatting:** at most 8 pages excluding references, single spaced, font size 10, 11 or 12, submitted as a PDF in one zip with the code, one submission per team.
- **Academic integrity:** contact the instructor before borrowing material when unsure.
- **Sample reports:** the project files include Sample 1 to 4 PDFs, probably example reports. They were not read for this handoff.

## 7. Technical facts already verified

The crash layer's schema and Hillsborough's 2017 class shares were checked against the live FDOT service on Oct 9; weather details and software versions were not.

**Crash layer:** Crashes\_All, layer 0. It covers 2011 to early 2019, so the complete years are 2012 to 2018. Coding shifted around 2014, which is why the study starts at 2014. The service returns at most 1,000 records per request, JSON only.

| Role | Field names |
| --- | --- |
| Label | `INJSEVER` |
| Time and place | `CRASH_DATE`, `CRASH_TIME`, `CALENDAR_YEAR`, `COUNTY_TXT`, plus three coordinate sets (officer-reported, FDOT-verified, basemap-snapped) whose exact names were not recorded |
| Conditions | `LGHT_COND_CD` (lighting), `EVNT_WTHR_COND_CD` (officer-reported weather), `RD_SRFC_COND_CD` (road surface) |
| Roadway | `SPEED_LIMIT`, `FUNCLASS`, `CNTOFLANES`, `GEO_URBAN_RURAL_IND`, `INTERSECTION_IND`, `AVERAGE_DAILY_TRAFFIC`, `IMPCT_TYP_CD` (manner of collision) |
| Leakage, exclude | `NUMBER_OF_INJURED`, `NUMBER_OF_KILLED`, `NUMBER_OF_SERIOUS_INJURIES`, `TOTAL_PERSONS`, the pre-built indicator flags |
| Roadway ID | Unknown. Not in the verified list, and the plan's traffic join depends on it |

**Hillsborough 2017:** 29,758 crashes; 1,243 severe (4.2%), 39% injury, 56% no injury. Statewide in 2017 there were only about 2,805 fatal crashes, which is why fatal is merged with incapacitating.

**Count sites:** 23 in Hillsborough. This is a current snapshot, so the 2014 to 2018 number may differ.

**GHCNh weather:**

- Available on AWS at `s3://noaa-ghcnh-pds` with no credentials, and as PSV or parquet files per station per year.
- Visibility is in km.
- Exact column names: unknown.
- Time convention: assumed UTC, not verified.

**Weather stations:** none chosen. Tampa International Airport was suggested; whether it has usable hourly data for 2014 to 2018 is unverified.

**Python and scikit-learn versions:** unknown.

## 8. Decisions and problems log

Three decisions date from Oct 9, and one conflict between the Oct 9 findings and the plan needs resolving before the weather join.

**Decisions (Oct 9):**

- Hillsborough County, 2014 to 2018. Train on 2014 to 2017, test on 2018.
- A 3-class label instead of binary.
- Counties are not pooled, because the severity mix varies widely between them, for example 19% injury crashes in Miami-Dade against 39% in Hillsborough.

**Time zone conflict:** on Oct 9 the layer metadata showed the crash date fields use a fixed Eastern Standard Time reference with no daylight saving. The plan's Phase 2 says to localize to `America/New_York`, which does apply daylight saving and would shift every summer crash by an hour against the weather data. Check a few known crash times in the actual records before choosing; until then, treat the plan's instruction as wrong.

**Unknown:** dropped-row counts, station distance cutoffs, and any other problems. Nothing has been recorded.

- [ ] Team to fill in: code location, workstream owners, the traffic data decision, Thanksgiving availability

## Sources

- Implementation Plan: Hillsborough County Crash Severity Prediction (Claude Doc, Nov 16)
- Oct 9 proposal chat, "Florida traffic crash prediction algorithm"
- Course Project Details sheet and the team's project proposal (project files)
