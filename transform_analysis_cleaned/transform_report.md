# Feature Transformation Recommendations

**Dataset:** `cleaned_lightpath_dataset.csv`

**Rows (full):** 835,269

**Rows (sampled for distribution stats):** 199,998

## How to read this

- `recommended_transform` is the default suggestion for gradient/distance-based models.
- Tree-based models generally do not require scaling; the main exception here is treating discrete features (`freq`, modulation orders, line rates) as categorical rather than continuous.
- For BER-like columns, scaling alone is not enough; a `log10` transform is typically the important step.

## Recommended transforms (per feature)

| feature | family | nunique_sample | min_full | max_full | skew_sample | outlier_rate_iqr_sample | recommended_transform |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avg_ber | ber | 177567 | 0 | 0.00352082 | -0.354267 | 0.0245552 | -log10(x+1e-12) then standardize |
| max_ber | ber | 33683 | 0 | 0.0038 | -4.1179 | 0.102826 | -log10(x+1e-12) then standardize |
| max_ber_left | ber | 172553 | 0 | 0.0038 | 0.445709 | 0 | -log10(x+1e-12) then standardize |
| max_ber_right | ber | 130507 | 0 | 0.0038 | 0.762307 | 0 | -log10(x+1e-12) then standardize |
| min_ber | ber | 50659 | 0 | 0.00352082 | 6.16673 | 0.0231452 | -log10(x+1e-12) then standardize |
| min_ber_left | ber | 173512 | 0 | 0.00379998 | 0.9373 | 0 | -log10(x+1e-12) then standardize |
| min_ber_right | ber | 131236 | 0 | 0.0038 | 1.31287 | 0.0418254 | -log10(x+1e-12) then standardize |
| num_links | counts | 9 | 1 | 9 | 0.223459 | 0.00378504 | standardize |
| num_spans | counts | 19 | 2 | 21 | 0.172899 | 0.000245002 | standardize |
| dst_degree | degree | 3 | 3 | 5 | 0.140779 | 0 | standardize |
| src_degree | degree | 3 | 3 | 5 | -0.277419 | 0 | standardize |
| freq | grid_frequency | 96 | 192.2 | 195.762 | 0.81297 | 0.00130001 | one-hot encode (categorical) |
| avg_link_len | length | 176 | 84 | 209 | 0.868299 | 0.0547855 | standardize |
| max_link_len | length | 26 | 84 | 313 | 0.49066 | 6.00006e-05 | standardize |
| min_link_len | length | 25 | 52 | 193 | 0.434995 | 0.206622 | standardize |
| path_len | length | 181 | 84 | 1382 | 0.186501 | 0.0102251 | standardize |
| conn_linerate | line_rate | 3 | 112 | 448 | 0.521464 | 0 | one-hot encode |
| lp_linerate | line_rate | 5 | 112 | 336 | 0.407506 | 0 | one-hot encode |
| max_lp_linerate_left | line_rate | 6 | 0 | 336 | -0.159717 | 0 | one-hot encode |
| max_lp_linerate_right | line_rate | 6 | 0 | 336 | -0.0997268 | 0 | one-hot encode |
| min_lp_linerate_left | line_rate | 6 | 0 | 336 | 1.08161 | 0.0262353 | one-hot encode |
| min_lp_linerate_right | line_rate | 6 | 0 | 336 | 1.59117 | 0.0122701 | one-hot encode |
| max_mod_order_left | mod_order | 4 | 0 | 64 | 0.0401669 | 0 | one-hot encode (including 0 if present) |
| max_mod_order_right | mod_order | 4 | 0 | 64 | 0.10358 | 0 | one-hot encode (including 0 if present) |
| min_mod_order_left | mod_order | 4 | 0 | 64 | 1.0967 | 0 | one-hot encode (including 0 if present) |
| min_mod_order_right | mod_order | 4 | 0 | 64 | 1.66191 | 0.0799958 | one-hot encode (including 0 if present) |
| mod_order | mod_order | 3 | 16 | 64 | -0.208062 | 0 | one-hot encode (including 0 if present) |
| avg_link_occ | occupancy | 1574 | 1 | 96 | -0.0736546 | 0.000675007 | standardize |
| max_link_occ | occupancy | 96 | 1 | 96 | -0.304825 | 0 | standardize |
| min_link_occ | occupancy | 96 | 1 | 96 | 0.222576 | 0.00351004 | standardize |
| std_link_occ | occupancy | 16992 | 0 | 40.025 | 0.486025 | 0.0102851 | standardize |
| sum_link_occ | occupancy | 533 | 1 | 551 | 0.547344 | 0.00119501 | standardize |

## Detailed per-feature justification

### ber

**avg_ber**

- Range (full): 0 .. 0.00352082
- Skew (sample): -0.354, outliers(IQR)≈2.46%
- Unique (sample): 177567, integer-like: 0.000
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `avg_ber` is BER-like (min=0, max=0.00352, zero%=0.03, skew≈-0.35). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

**max_ber**

- Range (full): 0 .. 0.0038
- Skew (sample): -4.118, outliers(IQR)≈10.28%
- Unique (sample): 33683, integer-like: 0.000
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `max_ber` is BER-like (min=0, max=0.0038, zero%=0.03, skew≈-4.12). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

**max_ber_left**

- Range (full): 0 .. 0.0038
- Skew (sample): 0.446, outliers(IQR)≈0.00%
- Unique (sample): 172553, integer-like: 0.032
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `max_ber_left` is BER-like (min=0, max=0.0038, zero%=3.25, skew≈0.45). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

**max_ber_right**

- Range (full): 0 .. 0.0038
- Skew (sample): 0.762, outliers(IQR)≈0.00%
- Unique (sample): 130507, integer-like: 0.272
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `max_ber_right` is BER-like (min=0, max=0.0038, zero%=27.15, skew≈0.76). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

**min_ber**

- Range (full): 0 .. 0.00352082
- Skew (sample): 6.167, outliers(IQR)≈2.31%
- Unique (sample): 50659, integer-like: 0.000
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `min_ber` is BER-like (min=0, max=0.00352, zero%=0.03, skew≈6.17). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

**min_ber_left**

- Range (full): 0 .. 0.00379998
- Skew (sample): 0.937, outliers(IQR)≈0.00%
- Unique (sample): 173512, integer-like: 0.032
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `min_ber_left` is BER-like (min=0, max=0.0038, zero%=3.25, skew≈0.94). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

**min_ber_right**

- Range (full): 0 .. 0.0038
- Skew (sample): 1.313, outliers(IQR)≈4.18%
- Unique (sample): 131236, integer-like: 0.272
- Recommended: -log10(x+1e-12) then standardize
- Alternative: -log10(x+1e-12) then min-max scale
- Why: `min_ber_right` is BER-like (min=0, max=0.0038, zero%=27.15, skew≈1.31). BERs span orders of magnitude; applying `-log10(x+1e-12)` spreads small values and makes this feature numerically well-scaled. Then standardize for models sensitive to feature scale (linear models, kNN, neural nets).

### counts

**num_links**

- Range (full): 1 .. 9
- Skew (sample): 0.223, outliers(IQR)≈0.38%
- Unique (sample): 9, integer-like: 1.000
- Recommended: standardize
- Alternative: one-hot encode
- Why: `num_links` is a low-cardinality integer count (nunique≈9). You can keep it numeric (scaled) or one-hot encode if you suspect non-linear jumps.

**num_spans**

- Range (full): 2 .. 21
- Skew (sample): 0.173, outliers(IQR)≈0.02%
- Unique (sample): 19, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `num_spans` is count-like; standardization keeps it comparable to other features for gradient-based models.

### degree

**dst_degree**

- Range (full): 3 .. 5
- Skew (sample): 0.141, outliers(IQR)≈0.00%
- Unique (sample): 3, integer-like: 1.000
- Recommended: standardize
- Alternative: one-hot encode
- Why: `dst_degree` is a low-cardinality integer count (nunique≈3). You can keep it numeric (scaled) or one-hot encode if you suspect non-linear jumps.

**src_degree**

- Range (full): 3 .. 5
- Skew (sample): -0.277, outliers(IQR)≈0.00%
- Unique (sample): 3, integer-like: 1.000
- Recommended: standardize
- Alternative: one-hot encode
- Why: `src_degree` is a low-cardinality integer count (nunique≈3). You can keep it numeric (scaled) or one-hot encode if you suspect non-linear jumps.

### grid_frequency

**freq**

- Range (full): 192.2 .. 195.762
- Skew (sample): 0.813, outliers(IQR)≈0.13%
- Unique (sample): 96, integer-like: 0.008
- Recommended: one-hot encode (categorical)
- Alternative: standardize (if forcing numeric)
- Why: `freq` has low-ish cardinality on a fixed grid (nunique≈96); treating it as continuous implies a linear ordering that usually does not reflect channel identity. Prefer categorical encoding.

### length

**avg_link_len**

- Range (full): 84 .. 209
- Skew (sample): 0.868, outliers(IQR)≈5.48%
- Unique (sample): 176, integer-like: 0.264
- Recommended: standardize
- Alternative: min-max scale
- Why: `avg_link_len` is continuous length-like; standardization is a solid default across many models.

**max_link_len**

- Range (full): 84 .. 313
- Skew (sample): 0.491, outliers(IQR)≈0.01%
- Unique (sample): 26, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `max_link_len` is continuous length-like; standardization is a solid default across many models.

**min_link_len**

- Range (full): 52 .. 193
- Skew (sample): 0.435, outliers(IQR)≈20.66%
- Unique (sample): 25, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `min_link_len` is continuous length-like; standardization is a solid default across many models.

**path_len**

- Range (full): 84 .. 1382
- Skew (sample): 0.187, outliers(IQR)≈1.02%
- Unique (sample): 181, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `path_len` is continuous length-like; standardization is a solid default across many models.

### line_rate

**conn_linerate**

- Range (full): 112 .. 448
- Skew (sample): 0.521, outliers(IQR)≈0.00%
- Unique (sample): 3, integer-like: 1.000
- Recommended: one-hot encode
- Alternative: standardize numeric
- Why: `conn_linerate` is a (discrete) line-rate-like feature (nunique≈3, integer-like=1.000). Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates.

**lp_linerate**

- Range (full): 112 .. 336
- Skew (sample): 0.408, outliers(IQR)≈0.00%
- Unique (sample): 5, integer-like: 1.000
- Recommended: one-hot encode
- Alternative: standardize numeric
- Why: `lp_linerate` is a (discrete) line-rate-like feature (nunique≈5, integer-like=1.000). Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates.

**max_lp_linerate_left**

- Range (full): 0 .. 336
- Skew (sample): -0.160, outliers(IQR)≈0.00%
- Unique (sample): 6, integer-like: 1.000
- Recommended: one-hot encode
- Alternative: standardize numeric
- Why: `max_lp_linerate_left` is a (discrete) line-rate-like feature (nunique≈6, integer-like=1.000). Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates.

**max_lp_linerate_right**

- Range (full): 0 .. 336
- Skew (sample): -0.100, outliers(IQR)≈0.00%
- Unique (sample): 6, integer-like: 1.000
- Recommended: one-hot encode
- Alternative: standardize numeric
- Why: `max_lp_linerate_right` is a (discrete) line-rate-like feature (nunique≈6, integer-like=1.000). Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates.

**min_lp_linerate_left**

- Range (full): 0 .. 336
- Skew (sample): 1.082, outliers(IQR)≈2.62%
- Unique (sample): 6, integer-like: 1.000
- Recommended: one-hot encode
- Alternative: standardize numeric
- Why: `min_lp_linerate_left` is a (discrete) line-rate-like feature (nunique≈6, integer-like=1.000). Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates.

**min_lp_linerate_right**

- Range (full): 0 .. 336
- Skew (sample): 1.591, outliers(IQR)≈1.23%
- Unique (sample): 6, integer-like: 1.000
- Recommended: one-hot encode
- Alternative: standardize numeric
- Why: `min_lp_linerate_right` is a (discrete) line-rate-like feature (nunique≈6, integer-like=1.000). Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates.

### mod_order

**max_mod_order_left**

- Range (full): 0 .. 64
- Skew (sample): 0.040, outliers(IQR)≈0.00%
- Unique (sample): 4, integer-like: 1.000
- Recommended: one-hot encode (including 0 if present)
- Alternative: map to bits_per_symbol=log2(x) then standardize
- Why: `max_mod_order_left` is modulation order-like (nunique≈4, integer-like=1.000). These values are discrete; either one-hot encode or map to an ordinal domain feature `log2(mod_order)` when >0 (bits per symbol).

**max_mod_order_right**

- Range (full): 0 .. 64
- Skew (sample): 0.104, outliers(IQR)≈0.00%
- Unique (sample): 4, integer-like: 1.000
- Recommended: one-hot encode (including 0 if present)
- Alternative: map to bits_per_symbol=log2(x) then standardize
- Why: `max_mod_order_right` is modulation order-like (nunique≈4, integer-like=1.000). These values are discrete; either one-hot encode or map to an ordinal domain feature `log2(mod_order)` when >0 (bits per symbol).

**min_mod_order_left**

- Range (full): 0 .. 64
- Skew (sample): 1.097, outliers(IQR)≈0.00%
- Unique (sample): 4, integer-like: 1.000
- Recommended: one-hot encode (including 0 if present)
- Alternative: map to bits_per_symbol=log2(x) then standardize
- Why: `min_mod_order_left` is modulation order-like (nunique≈4, integer-like=1.000). These values are discrete; either one-hot encode or map to an ordinal domain feature `log2(mod_order)` when >0 (bits per symbol).

**min_mod_order_right**

- Range (full): 0 .. 64
- Skew (sample): 1.662, outliers(IQR)≈8.00%
- Unique (sample): 4, integer-like: 1.000
- Recommended: one-hot encode (including 0 if present)
- Alternative: map to bits_per_symbol=log2(x) then standardize
- Why: `min_mod_order_right` is modulation order-like (nunique≈4, integer-like=1.000). These values are discrete; either one-hot encode or map to an ordinal domain feature `log2(mod_order)` when >0 (bits per symbol).

**mod_order**

- Range (full): 16 .. 64
- Skew (sample): -0.208, outliers(IQR)≈0.00%
- Unique (sample): 3, integer-like: 1.000
- Recommended: one-hot encode (including 0 if present)
- Alternative: map to bits_per_symbol=log2(x) then standardize
- Why: `mod_order` is modulation order-like (nunique≈3, integer-like=1.000). These values are discrete; either one-hot encode or map to an ordinal domain feature `log2(mod_order)` when >0 (bits per symbol).

### occupancy

**avg_link_occ**

- Range (full): 1 .. 96
- Skew (sample): -0.074, outliers(IQR)≈0.07%
- Unique (sample): 1574, integer-like: 0.349
- Recommended: standardize
- Alternative: min-max scale
- Why: `avg_link_occ` appears reasonably well-behaved (skew≈-0.07, outliers≈0.07%). Standardization is a good default.

**max_link_occ**

- Range (full): 1 .. 96
- Skew (sample): -0.305, outliers(IQR)≈0.00%
- Unique (sample): 96, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `max_link_occ` appears reasonably well-behaved (skew≈-0.30, outliers≈0.00%). Standardization is a good default.

**min_link_occ**

- Range (full): 1 .. 96
- Skew (sample): 0.223, outliers(IQR)≈0.35%
- Unique (sample): 96, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `min_link_occ` appears reasonably well-behaved (skew≈0.22, outliers≈0.35%). Standardization is a good default.

**std_link_occ**

- Range (full): 0 .. 40.025
- Skew (sample): 0.486, outliers(IQR)≈1.03%
- Unique (sample): 16992, integer-like: 0.169
- Recommended: standardize
- Alternative: min-max scale
- Why: `std_link_occ` appears reasonably well-behaved (skew≈0.49, outliers≈1.03%). Standardization is a good default.

**sum_link_occ**

- Range (full): 1 .. 551
- Skew (sample): 0.547, outliers(IQR)≈0.12%
- Unique (sample): 533, integer-like: 1.000
- Recommended: standardize
- Alternative: min-max scale
- Why: `sum_link_occ` appears reasonably well-behaved (skew≈0.55, outliers≈0.12%). Standardization is a good default.

