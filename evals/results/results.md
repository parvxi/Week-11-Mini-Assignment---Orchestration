| Case | Measures | Threshold | Baseline 1 · fixed Airflow workflow | Baseline 2 · class agent loop | Part 2 loop (this repo) | Part 3 adaptive system |
| --- | --- | ---: | --- | --- | --- | --- |
| `outcome_columns_excluded` | intermediate | 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 | ✅ 1.00 |
| `leak_disguised` | intermediate | 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 | ✅ 1.00 |
| `no_false_block` | intermediate | 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| `probe_accuracy` | intermediate | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `forecast_skill` | final | 0.90 | ❌ 0.82 | ❌ 0.79 | ✅ 0.97 | ❌ 0.89 |
| `ranking_quality` | final | 0.75 | ❌ 0.72 | ❌ 0.72 | ✅ 0.81 | ✅ 0.79 |
| `report_matches_referee` | final | 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 | ✅ 1.00 |
| `reports_no_signal` | final | 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 | ✅ 1.00 |
| `dirty_data_recovered` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `repair_routing` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `invalid_data_stops` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `efficiency_no_signal` | coordination | 0.50 | ✅ 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `plateau_stop` | coordination | 1.00 | ✅ 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `fault_transient` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 | ✅ 1.00 |
| `fault_family_outage` | coordination | 1.00 | ❌ 0.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| `fault_total_outage` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `retry_limit` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `agent_selection` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| `plan_integrity` | coordination | 1.00 | ❌ 0.00 | ❌ 0.00 | ❌ 0.00 | ✅ 1.00 |
| **Passed** | | | **3/19** | **2/19** | **9/19** | **18/19** |
