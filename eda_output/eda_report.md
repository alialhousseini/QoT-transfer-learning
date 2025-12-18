# Lightpath QoT EDA Report

**Dataset:** `cleaned_lightpath_dataset.csv`

## 1) Overview

| Metric | Value |
|---|---|
| Total samples | 835,269 |
| Features | 32 |
| Target metrics | 1 |
| Analyzed rows | 835,269 |
| Analyzed fraction | 100.000% |
| Pandas `data` size | 203.9 MB |
| Pandas `target` size | 6.4 MB |

**Features (35):**

path_len, avg_link_len, min_link_len, max_link_len, num_links, num_spans, freq, mod_order, lp_linerate, conn_linerate, src_degree, dst_degree, sum_link_occ, min_link_occ, max_link_occ, avg_link_occ, std_link_occ, max_ber, min_ber, avg_ber, min_mod_order_left, max_mod_order_left, min_mod_order_right, max_mod_order_right, min_lp_linerate_left, max_lp_linerate_left, min_lp_linerate_right, max_lp_linerate_right, min_ber_left, max_ber_left, min_ber_right, max_ber_right

**Targets (4):**

class

## 2) Missing Values (on analyzed rows)

### Data

| index | missing |
| --- | --- |
| path_len | 0 |
| avg_link_len | 0 |
| min_ber_right | 0 |
| max_ber_left | 0 |
| min_ber_left | 0 |
| max_lp_linerate_right | 0 |
| min_lp_linerate_right | 0 |
| max_lp_linerate_left | 0 |
| min_lp_linerate_left | 0 |
| max_mod_order_right | 0 |
| min_mod_order_right | 0 |
| max_mod_order_left | 0 |
| min_mod_order_left | 0 |
| avg_ber | 0 |
| min_ber | 0 |
| max_ber | 0 |
| std_link_occ | 0 |
| avg_link_occ | 0 |
| max_link_occ | 0 |
| min_link_occ | 0 |

### Target

| index | missing |
| --- | --- |
| class | 0 |

## 3) Descriptive Statistics (on analyzed rows)

### Key feature stats (selected columns)

| index | count | mean | std | min | 1% | 5% | 25% | 50% | 75% | 95% | 99% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| path_len | 835269.000000 | 517.655670 | 217.255528 | 84.000000 | 99.000000 | 143.000000 | 368.000000 | 505.000000 | 652.000000 | 888.000000 | 1028.000000 | 1382.000000 |
| avg_link_len | 835269.000000 | 135.487610 | 22.174926 | 84.000000 | 99.000000 | 108.000000 | 119.500000 | 129.400000 | 144.333000 | 184.000000 | 193.000000 | 209.000000 |
| min_link_len | 835269.000000 | 101.616319 | 25.373692 | 52.000000 | 52.000000 | 52.000000 | 89.000000 | 101.000000 | 107.000000 | 154.000000 | 157.000000 | 193.000000 |
| max_link_len | 835269.000000 | 177.389811 | 42.724862 | 84.000000 | 99.000000 | 114.000000 | 154.000000 | 156.000000 | 214.000000 | 267.000000 | 267.000000 | 313.000000 |
| num_links | 835269.000000 | 3.895746 | 1.714004 | 1.000000 | 1.000000 | 1.000000 | 3.000000 | 4.000000 | 5.000000 | 7.000000 | 8.000000 | 9.000000 |
| num_spans | 835269.000000 | 8.281846 | 3.529185 | 2.000000 | 2.000000 | 2.000000 | 6.000000 | 8.000000 | 11.000000 | 14.000000 | 17.000000 | 21.000000 |
| freq | 835269.000000 | 193.257375 | 0.870854 | 192.200000 | 192.200000 | 192.237500 | 192.537500 | 193.025000 | 193.812500 | 194.975000 | 195.537500 | 195.762500 |
| sum_link_occ | 835269.000000 | 174.305128 | 106.288812 | 1.000000 | 10.000000 | 31.000000 | 86.000000 | 160.000000 | 248.000000 | 368.000000 | 442.000000 | 551.000000 |
| avg_link_occ | 835269.000000 | 43.278741 | 16.346743 | 1.000000 | 6.000000 | 15.667000 | 31.857000 | 43.500000 | 55.250000 | 69.500000 | 77.333000 | 96.000000 |
| std_link_occ | 835269.000000 | 10.643225 | 7.072092 | 0.000000 | 0.000000 | 0.000000 | 5.535000 | 10.171000 | 14.872000 | 24.074000 | 28.897320 | 40.025000 |
| avg_ber | 835269.000000 | 0.001339 | 0.000330 | 0.000000 | 0.000482 | 0.000778 | 0.001156 | 0.001350 | 0.001544 | 0.001847 | 0.002120 | 0.003521 |

### Target metric stats

| index | count | mean | std | min | 1% | 5% | 25% | 50% | 75% | 95% | 99% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class | 835269.000000 | 0.718296 | 0.449830 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## 4) Discrete / Categorical-Like Columns

Detected discrete-like columns (<= 50 unique values on analyzed rows, plus a few forced IDs/rates):

path_len, min_link_len, max_link_len, num_links, num_spans, mod_order, lp_linerate, conn_linerate, src_degree, dst_degree, min_link_occ, max_link_occ, min_mod_order_left, max_mod_order_left, min_mod_order_right, max_mod_order_right, min_lp_linerate_left, max_lp_linerate_left, min_lp_linerate_right, max_lp_linerate_right

### `path_len` top-3

| path_len | count |
| --- | --- |
| 467.000000 | 24101.000000 |
| 510.000000 | 18574.000000 |
| 709.000000 | 17970.000000 |

### `min_link_len` top-3

| min_link_len | count |
| --- | --- |
| 102.000000 | 174869.000000 |
| 89.000000 | 120461.000000 |
| 99.000000 | 109520.000000 |

### `max_link_len` top-3

| max_link_len | count |
| --- | --- |
| 154.000000 | 124455.000000 |
| 194.000000 | 106108.000000 |
| 223.000000 | 101421.000000 |

### `num_links` top-3

| num_links | count |
| --- | --- |
| 4.000000 | 189867.000000 |
| 3.000000 | 158168.000000 |
| 5.000000 | 151319.000000 |

### `num_spans` top-3

| num_spans | count |
| --- | --- |
| 8.000000 | 136125.000000 |
| 7.000000 | 83590.000000 |
| 6.000000 | 80175.000000 |

### `mod_order` top-3

| mod_order | count |
| --- | --- |
| 64.000000 | 430906.000000 |
| 32.000000 | 362243.000000 |
| 16.000000 | 42120.000000 |

### `lp_linerate` top-3

| lp_linerate | count |
| --- | --- |
| 112.000000 | 336891.000000 |
| 224.000000 | 308360.000000 |
| 336.000000 | 77325.000000 |

### `conn_linerate` top-3

| conn_linerate | count |
| --- | --- |
| 112.000000 | 293774.000000 |
| 224.000000 | 293120.000000 |
| 448.000000 | 248375.000000 |

### `src_degree` top-3

| src_degree | count |
| --- | --- |
| 5.000000 | 334930.000000 |
| 4.000000 | 295762.000000 |
| 3.000000 | 204577.000000 |

### `dst_degree` top-3

| dst_degree | count |
| --- | --- |
| 3.000000 | 334328.000000 |
| 5.000000 | 273025.000000 |
| 4.000000 | 227916.000000 |

### `min_link_occ` top-3

| min_link_occ | count |
| --- | --- |
| 1.000000 | 20763.000000 |
| 32.000000 | 18816.000000 |
| 33.000000 | 18783.000000 |

### `max_link_occ` top-3

| max_link_occ | count |
| --- | --- |
| 53.000000 | 15425.000000 |
| 54.000000 | 15425.000000 |
| 55.000000 | 15267.000000 |

### `min_mod_order_left` top-3

| min_mod_order_left | count |
| --- | --- |
| 0.000000 | 507902.000000 |
| 32.000000 | 180905.000000 |
| 64.000000 | 110694.000000 |

### `max_mod_order_left` top-3

| max_mod_order_left | count |
| --- | --- |
| 32.000000 | 442376.000000 |
| 64.000000 | 302502.000000 |
| 16.000000 | 63376.000000 |

### `min_mod_order_right` top-3

| min_mod_order_right | count |
| --- | --- |
| 0.000000 | 604051.000000 |
| 32.000000 | 137531.000000 |
| 64.000000 | 66873.000000 |

### `max_mod_order_right` top-3

| max_mod_order_right | count |
| --- | --- |
| 32.000000 | 340889.000000 |
| 0.000000 | 225764.000000 |
| 64.000000 | 222495.000000 |

### `min_lp_linerate_left` top-3

| min_lp_linerate_left | count |
| --- | --- |
| 0.000000 | 507902.000000 |
| 112.000000 | 149086.000000 |
| 224.000000 | 106862.000000 |

### `max_lp_linerate_left` top-3

| max_lp_linerate_left | count |
| --- | --- |
| 224.000000 | 328882.000000 |
| 112.000000 | 262862.000000 |
| 280.000000 | 88180.000000 |

### `min_lp_linerate_right` top-3

| min_lp_linerate_right | count |
| --- | --- |
| 0.000000 | 604051.000000 |
| 112.000000 | 111740.000000 |
| 224.000000 | 79586.000000 |

### `max_lp_linerate_right` top-3

| max_lp_linerate_right | count |
| --- | --- |
| 224.000000 | 251657.000000 |
| 0.000000 | 225764.000000 |
| 112.000000 | 190892.000000 |

## 5) Consistency / Sanity Checks (on analyzed rows)

| Check | Violations |
|---|---|
| avg_link_len_outside_minmax | 0 (0.00%) |
| path_len_inconsistent_num_links_avg_link_len | 0 (0.00%) |
| num_spans_leq_num_links | 0 (0.00%) |
| avg_ber_outside_minmax | 0 (0.00%) |
| sum_link_occ_inconsistent_avg_link_occ_num_links | 0 (0.00%) |
| freq_off_inferred_grid_tol | 0 (0.00%) |
| class_nan | 0 (0.00%) |
| class_non_binary | 0 (0.00%) |

Interpretation notes:
- Some checks are strict equalities/tolerances and may flag harmless floating-point effects.
- If `freq_off_inferred_grid_tol` is high, consider snapping `freq` to the nearest inferred grid.

Inferred frequency grid:
- spacing: `0.037500` THz
- anchor(min): `192.200000` THz
- tol: `0.00003750` THz

## 6) Correlations (on analyzed rows)

- Heatmap: `eda_output/figures/corr_heatmap_features.png`

### Strongest feature-feature correlations (|r| ≥ 0.75)

| a | b | corr |
| --- | --- | --- |
| num_spans | path_len | 0.988579 |
| num_links | num_spans | 0.971141 |
| path_len | num_links | 0.954119 |
| max_link_occ | avg_link_occ | 0.863418 |
| min_lp_linerate_right | min_mod_order_right | 0.832310 |
| min_link_occ | avg_link_occ | 0.824625 |
| min_mod_order_left | min_lp_linerate_left | 0.806418 |
| num_links | sum_link_occ | 0.803648 |
| max_link_occ | sum_link_occ | 0.799867 |
| min_ber_right | max_ber_right | 0.790053 |
| sum_link_occ | num_spans | 0.760892 |
| min_ber_left | max_ber_left | 0.758359 |

### Feature ↔ target correlations (Pearson)

| index | class |
| --- | --- |
| avg_ber | -0.062522 |
| avg_link_len | 0.075531 |
| avg_link_occ | -0.002325 |
| conn_linerate | 0.119270 |
| dst_degree | 0.034613 |
| freq | -0.157524 |
| lp_linerate | -0.072299 |
| max_ber | -0.148429 |
| max_ber_left | -0.092757 |
| max_ber_right | -0.076312 |
| max_link_len | -0.099506 |
| max_link_occ | -0.165249 |
| max_lp_linerate_left | -0.031623 |
| max_lp_linerate_right | -0.052149 |
| max_mod_order_left | 0.058751 |
| max_mod_order_right | 0.002190 |
| min_ber | 0.092962 |
| min_ber_left | -0.018946 |
| min_ber_right | -0.017927 |
| min_link_len | 0.230618 |
| min_link_occ | 0.146654 |
| min_lp_linerate_left | 0.196211 |
| min_lp_linerate_right | 0.131777 |
| min_mod_order_left | 0.234570 |
| min_mod_order_right | 0.166314 |
| mod_order | -0.287300 |
| num_links | -0.362554 |
| num_spans | -0.357477 |
| path_len | -0.353628 |
| src_degree | -0.058933 |
| std_link_occ | -0.266661 |
| sum_link_occ | -0.222063 |

## 7) Target Class Balance (on analyzed rows)

| class | count | pct |
| --- | --- | --- |
| 1.000000 | 599970.000000 | 0.718296 |
| 0.000000 | 235299.000000 | 0.281704 |

Interpretation notes:
- If `class` is imbalanced, prefer stratified splits and metrics like PR-AUC/F1.

## 8) Figures

- `eda_output/figures/hist_path_len.png`
- `eda_output/figures/hist_avg_link_len.png`
- `eda_output/figures/hist_min_link_len.png`
- `eda_output/figures/hist_max_link_len.png`
- `eda_output/figures/hist_num_links.png`
- `eda_output/figures/hist_num_spans.png`
- `eda_output/figures/hist_freq.png`
- `eda_output/figures/hist_sum_link_occ.png`
- `eda_output/figures/hist_avg_link_occ.png`
- `eda_output/figures/hist_std_link_occ.png`
- `eda_output/figures/hist_avg_ber.png`

## 9) Practical Next Steps

- Decide whether IDs (`conn_id`, `src_id`, `dst_id`) are allowed features; often they should be dropped.
- Consider modeling `freq`, `mod_order`, `lp_linerate`, `conn_linerate` as categorical/discrete.
- If sanity-check violations are common, decide whether to filter rows, fix by rule, or add features capturing the inconsistency.
- For prediction targets (`osnr`, `snr`, `ber`), check skew and consider log-transforming `ber`.

