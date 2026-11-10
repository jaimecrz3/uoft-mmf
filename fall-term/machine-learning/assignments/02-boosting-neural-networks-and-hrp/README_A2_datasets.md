# MMF2030H — Assignment 2 — Provided Datasets

Three fixed datasets are provided for Part B so everyone works from identical data with a checkable reference solution. Read this before starting B1–B3.

## mmf2030h_a2_credit_data.csv

8,000 borrowers, one row per borrower. Used for **B1** (XGBoost + SHAP) and **B2** (neural networks).

| Column | Description |
|---|---|
| `revolving_utilization` | Total balance on revolving credit lines (credit cards, lines of credit) divided by total credit limits across those lines. Can exceed 1.0 for an overutilized borrower. |
| `debt_ratio` | Monthly debt payments (including rent/mortgage) divided by gross monthly income. |
| `age` | Borrower age in years at time of application. |
| `monthly_income` | Gross monthly income, in dollars. |
| `num_past_due_30_59` | Number of times the borrower was 30–59 days past due on any payment in the past 2 years. |
| `num_past_due_60_89` | Number of times the borrower was 60–89 days past due on any payment in the past 2 years. |
| `num_open_credit_lines` | Number of open credit lines and loans. |
| `num_dependents` | Number of dependents. |
| `credit_bureau_score_z` | A standardized (z-scored) composite score sourced from a third-party credit bureau at time of application. |
| `application_channel_id` | Numeric identifier for the channel through which the application was submitted, encoded as a continuous value. |
| `serious_dlqin_2yrs` | **Target.** 1 if the borrower had a serious delinquency within 2 years of application, 0 otherwise. Base rate ≈ 10.3%. |
| `split` | `"train"` or `"test"` — the fixed evaluation split for this assignment. Use it exactly as given (see note below). |

## mmf2030h_a2_finbert_headlines.csv

2,100 short finance headlines, one row per headline. Not needed for required B2 (i–iii) — only for the two ungraded "Optional — More Practice" items at the end of B2: Hugging Face fine-tuning with LoRA, and the interpretability deep-dive (saliency maps / Integrated Gradients, compared against B1's SHAP values).

| Column | Description |
|---|---|
| `id` | Row identifier. |
| `headline` | Short synthetic finance headline (earnings, guidance, management changes, regulatory items, routine announcements). |
| `label` | **Target.** One of `positive`, `negative`, `neutral`. Balanced: 700 rows per class. |
| `split` | `"train"` or `"test"` — the fixed evaluation split (1,680 train / 420 test, 80/20). Use it exactly as given, same discipline as the credit dataset below. |

## mmf2030h_a2_prices.csv

Daily closing prices for 8 tickers, January 2020 – October 2024 (1,250 trading days). Used for **B3** (HRP vs. Markowitz).

- `TECH_A`, `TECH_B`, `TECH_C`, `TECH_D` — four names from one sector
- `STPL_A`, `STPL_B`, `STPL_C`, `STPL_D` — four names from a second sector

All eight series start at $100 on 2020-01-02 (indexed, not real quoted prices). Compute returns (simple or log) from these prices as your first step; use all 8 tickers and the full date range as instructed in B3.i.

## Train/test split — read this

Both the credit dataset and the headlines dataset already mark each row's evaluation role in their `split` column. **Train your B1/B2 models only on `split == "train"` rows, and report all performance metrics (AUC, F1, etc.) on `split == "test"` rows only.** Do not re-shuffle, re-split, or cross-validate across the whole file — the reference solution and the grading tolerance bands assume this exact split. Submissions that report metrics computed on a different split (or on the training set) will not match the reference numbers and may be marked down.

## A note on the data

All three datasets were constructed for this assignment rather than pulled from a public source. This means everyone works from identical, versioned data with a known reference solution — but these are illustrative datasets, not real borrowers, securities, or news headlines, so don't read anything into the specific values beyond what the assignment asks you to do with them. (The headlines dataset is template-generated with fictional company names, deliberately kept clean-labeled so the optional fine-tuning practice is learnable without specialized finance-NLP data.)
