# How the project and datasets were chosen

Checked on 12 November 2026. Every row count, schema and date range below was read from the
live source, not assumed.

## Portals tried

| Portal | Result |
| --- | --- |
| [KAPSARC Data Portal](https://datasource.kapsarc.org/) | Reachable. Opendatasoft API, 1,317 datasets, programmatic CSV export. Used. |
| [DataSaudi](https://datasaudi.sa/en/data-explorer/datasets) | Page loads (HTTP 200), but it is an interactive explorer; no bulk download endpoint was found in the time available. |
| [open.data.gov.sa](https://open.data.gov.sa/) | Connection timed out from this machine. Not used. |
| [stats.gov.sa](https://www.stats.gov.sa/) (GASTAT) | Reachable, but its tables that matter here are republished on KAPSARC with an API, so those copies were inspected instead. |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | Reachable. Serves Copernicus CAMS data. Used. |
| [NASA POWER](https://power.larc.nasa.gov/) | Reachable (daily MERRA-2 weather). Verified but not needed: the NOAA stations already report weather. |

## Candidates inspected

| # | Idea | Datasets actually inspected | Rows | What the data showed |
| --- | --- | --- | ---: | --- |
| A | **Next-day dust / low-visibility forecast at Saudi airports** | NOAA ISD hourly observations for Saudi stations (`saudi-hourly-weather-data` on KAPSARC, public domain) + CAMS dust, PM10, PM2.5 and aerosol optical depth at each station (Open-Meteo, CC BY 4.0) | 1,465,518 + 752,640 | 28 stations with about 33,000 hourly reports each from Aug 2022 to Aug 2025. Observed visibility gives a real label. The two sources join on station and day with no loss. |
| B | Riyadh air-quality forecasting | `air-quality` on KAPSARC (General Authority of Meteorology and Environmental Protection) + weather | 17,298 | One city, 2019–2020 only, and the licence field says "Restricted". Rejected on licence and coverage. |
| C | Labour-market indicators | `labor-force-survey-data` (GASTAT, 48,792 rows) + `labor-market-administrative-records-data` (31,146 rows) | 79,938 | Long-format published aggregates: one value per indicator, quarter and breakdown. Each series has only a few dozen quarters, which is too few points for a defensible train / validation / test split. |
| D | Road safety by region | `saudi-arabia-traffic-accidents-and-casualties-injured-dead-2008` (GASTAT) | 168 | Annual counts for 13 regions. Far too small for machine learning. |
| E | COVID-19 cases by city | `saudi-arabia-coronavirus-disease-covid-19-situation` (815,768 rows) + Google mobility (134 rows) | 815,902 | Large, but the second dataset is tiny and the question is no longer current. |

## Decision matrix

Scores are 1 (poor) to 5 (strong). Weights reflect what the assignment grades.

| Criterion | Weight | A dust | B air quality | C labour | D road safety | E COVID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Data accessible and reliable (API, licence) | 20% | 5 | 2 | 4 | 4 | 4 |
| Valid ML task (real label, enough rows, no built-in leak) | 20% | 4 | 3 | 2 | 1 | 3 |
| Fits Airflow requirements (two sources, per-source cleaning, merge) | 15% | 5 | 4 | 3 | 2 | 3 |
| Supports agent and bonus work (meaningful decisions, baselines) | 15% | 5 | 4 | 2 | 1 | 3 |
| Saudi relevance and interest | 10% | 5 | 4 | 5 | 4 | 2 |
| Reproducible | 10% | 5 | 2 | 4 | 4 | 3 |
| Feasible before the deadline | 5% | 4 | 3 | 3 | 5 | 3 |
| Portfolio value | 5% | 5 | 3 | 3 | 2 | 2 |
| **Weighted total** | | **4.75** | **3.10** | **3.15** | **2.60** | **3.05** |

**Chosen: A.** It is the only candidate with two independent, openly licensed sources, an
observed (not modelled) outcome, tens of thousands of rows, and a clean join key.

## Why this problem matters

Dust is Saudi Arabia's most frequent weather hazard. Low visibility delays and diverts flights,
closes roads, and sends people with asthma to hospital. An airport operations desk that knows
tonight that tomorrow is likely to be a low-visibility day can adjust schedules and staffing.

## Verified facts about the two chosen sources

| | NOAA ISD (via KAPSARC) | CAMS (via Open-Meteo) |
| --- | --- | --- |
| Original producer | NOAA National Centers for Environmental Information | Copernicus Atmosphere Monitoring Service (ECMWF) |
| Endpoint | `https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/saudi-hourly-weather-data/exports/csv` | `https://air-quality-api.open-meteo.com/v1/air-quality` |
| Format | CSV export, 33 columns (12 selected) | JSON per coordinate, hourly arrays |
| Rows downloaded | 948,244 hourly reports from 1 Aug 2022 | 752,640 station-hours (28 stations × 26,880 hours) |
| Date range | 2022-08-01 to 2025-08-24 | 2022-08-01 to 2025-08-24 |
| Coverage | 35 Saudi stations in the window, 28 with enough reports | The same 28 station coordinates (about 0.4° model grid) |
| Missing values | No nulls; missing readings are sentinel numbers (pressure is a sentinel in 66% of reports) | 0.24% null per variable |
| Duplicates | 216,109 duplicate station-hour reports | none |
| Join key | `station_id` + local calendar day | `station_id` + local calendar day |
| Licence | Public domain | CC BY 4.0 |

Limits worth knowing: the ISD copy on KAPSARC ends on 24 August 2025, and CAMS is a model
analysis on a coarse grid, not a measurement at the airport.
