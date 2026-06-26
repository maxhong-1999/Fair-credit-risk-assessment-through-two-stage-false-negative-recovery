# Fair-credit-risk-assessment-through-two-stage-false-negative-recovery

This repository is designed to ensure the reproducibility of the article. <br />
**Note**: This study was conducted with KCB (Korea Credit Bureau) Data.

---

## Abstract

This study addresses a practical tension in financial inclusion: how to preserve access while identifying high-risk borrowers who were initially overlooked by conventional credit scoring. It proposes a two-stage credit risk assessment framework in which false negatives (FN) are defined as high-risk borrowers misclassified as low risk. In this study, fairness is interpreted in a limited operational sense as data-level comparability across borrowers, rather than as formal algorithmic fairness across protected groups. Using a large-scale dataset from the Korea Credit Bureau (KCB), the analysis reconstructs credit records at the person level and applies a personalised parallel window to align observation periods before model training. The framework then combines broad first-stage risk screening with a focused post-hoc reassessment of previously overlooked cases using policy loan borrower data as a reference domain. Under this retrospective design, Stage 1 FNs declined from 469 to 321, and a proxy-based retrospective simulation within the Stage 1 FN subset indicated a 37.0% relative reduction in proxy-based loss estimates. These findings suggest that structured data alignment and post-hoc FN recovery can support model auditing, policy recalibration, and more consistent credit assessment, while not constituting evidence of realised institutional savings or formal group fairness.

---

## Two-stage risk management framework

This study uses policy loan borrower data as a policy-relevant complementary reference domain for post-hoc FN reassessment. This use does not assume equivalence between policy loan borrowers and FN cases; rather, it examines whether risk patterns observed in a vulnerable-borrower domain provide additional signals for reassessing overlooked high-risk borrowers.

- Stage 1 (G-model): A generalised prediction model applied to the entire borrower population to screen the overall credit risk structure.
- Stage 2 (P-model): A refined evaluation model that uses policy loan borrower data to re-evaluate cases later identified as false negatives based on realised outcomes.
  
This framework addresses a limitation of existing credit scoring research by separating broad risk screening from focused post-hoc reassessment. Stage 1 maintains coverage and efficiency across the overall borrower population, while Stage 2 provides a complementary reassessment of previously overlooked high-risk cases. Managing FNs as a distinct risk segment provides a structured analytical approach to examining overlooked risks and their implications for institutional loss, with the policy loan borrower domain serving as complementary information rather than a direct representation of FN cases.

---

## Limitations of traditional window-based modelling

The window-based approach has evolved from simple time-series segmentation to a core analytical structure for capturing temporal and contextual variations. However, using windows does not guarantee that all temporal characteristics are equally represented, as structural imbalances from differences in observation frequency and information volume persist.

<img width="5760" height="3240" alt="image" src="https://github.com/user-attachments/assets/c91af25d-0b5f-46c5-a084-8016d252d100" />
<p align='center'>Figure 1. Data distribution in traditional window</p>

Figure 1 illustrates how windowing includes or excludes samples. Figure 1(a) shows a snapshot window, while Figure 1(b) shows a rolling window. Blue areas represent sections included in the model input (Window), while orange areas indicate excluded sections (Out of Window). Each line represents the actual observation sequence of an individual data record: black segments denote observations included within the window and red segments indicate excluded portions. Even with identical window time spans, substantial differences remain in the number of included observations and information density across individuals. These differences introduce structural imbalances into the reliability and comparability of the summary statistics computed within the window.

</br>

<img width="5760" height="3240" alt="image" src="https://github.com/user-attachments/assets/ead598c9-fdee-4f39-a226-202f6136eddd" />
<p align='center'>Figure 2. Realignment process of personalised parallel windows</p>

As discussed earlier, conventional time-based window settings often produce data imbalance because the individual-level observation periods available vary. To address this, this study proposes the personalised parallel window technique. As illustrated in Figure 2, each individual’s most recent observation point was set as the reference point, and a fixed preceding period defined the start of the observation window, establishing an identical individual-specific window. Records for which a consistent-length observed window could not be obtained within an individual’s available data were excluded to maintain informational balance across the dataset.

---

## Method of Our Research

<img width="5760" height="1620" alt="image" src="https://github.com/user-attachments/assets/6072dd62-4609-46a3-b59e-ab7ccc38614c" />
<p align='center'>Figure 3. FN recovery framework</p>

Figure 3 illustrates the overall structure of the framework applying the algorithms. This study designed a two-stage risk management framework to improve the precision of post-hoc FN re-evaluation. In Stage 1, a general model (G-model) was trained using the entire borrower dataset. In Stage 2, a specialised model (P-model) trained on policy loan borrower data was applied to the FNs identified by the G-model. This structure reflects the characteristics of policy loan borrowers—the primary targets of policy support—while re-evaluating all FN cases to recover previously overlooked high-risk cases.</br>

---

## Datasets

### Repository Structure

All code requires the corresponding YAML files in `configs/`. The paths are as follows:

```
configs/
  paths.yaml              Local path configuration
  model_config.yaml       Model and validation settings
data/
  README.md               Explains restricted-data placement, no data included
src/
  DataPreprocess/
    data_preprocess.py
    cohort_filtering.py
  FeatureEngineering/
    feature_engineering.py
    feature_selection.py
  Experiments/
    vanilla_model.py
    hyper_parameter_model.py
    two_stage_evaluation.py
  Discussion/
    proxy_loss.py
```

---

### Dataset Statistics

The data used in this study were provided by the KCB under a research agreement and cannot be publicly shared due to confidentiality and legal restrictions. Supplementary summary tables and methodological materials are provided with this article. The underlying raw and cleaned data are not publicly available because they are subject to third-party data-use restrictions. However, researchers who require access to the data may contact the corresponding author, and data may be made available upon reasonable request, subject to approval by the data provider and applicable restrictions.

#### Dataset Columns

| No. | Column | Description | Type | Notes |
|----:|---------|-------------|------|-------|
| 1 | KCB_DEID1_ENCRYPT | Customer identifier (encrypted) | Float | - |
| 2 | PERIOD | Observation period | categorical | Survey reference period |
| 3 | GENDER | Gender | categorical | 1, 2 (latest) |
| 4 | AGE_CD | Age group | categorical | 20, 30, ..., 70 (latest) |
| 5 | JOB_CD | Occupation category | categorical | Latest |
| 6 | JOB_MOVE_YN | Occupation change | binary | 1=yes, 0=no (both periods) |
| 7 | HOM_CD | Residential area code | categorical | Latest |
| 8 | HOM_MOVE_YN | Residential move | binary | 1=yes, 0=no (both periods) |
| 9 | COM_CD | Workplace area code | categorical | Latest |
| 10 | COM_MOVE_YN | Workplace change | binary | 1=yes, 0=no (both periods) |
| 11 | INCOM_AVG | Average income | numeric | Both periods |
| 12 | INCOM_LAST | Latest income | numeric | Latest |
| 13 | INCOM_SLOPE | Income change | numeric | Trend |
| 14 | SCORE_AVG | Average credit score | numeric | Both periods |
| 15 | SCORE_LAST | Latest credit score | numeric | Latest |
| 16 | SCORE_SLOPE | Credit score change | numeric | Both periods |
| 17 | LN_SUN_AVG | Average Sunshine Loan balance | numeric | Both periods |
| 18 | LN_SUN_LAST | Latest Sunshine Loan balance | numeric | Latest |
| 19 | LN_SUN_SLOPE | Sunshine Loan balance change | numeric | Trend |
| 20 | LN_NHP_AVG | Average New Hope Loan balance | numeric | Both periods |
| 21 | LN_NHP_LAST | Latest New Hope Loan balance | numeric | Latest |
| 22 | LN_NHP_SLOPE | New Hope Loan balance change | numeric | Trend |
| 23 | LN_ETC_AVG | Average balance of other policy loans | numeric | Both periods |
| 24 | LN_ETC_LAST | Latest balance of other policy loans | numeric | Latest |
| 25 | LN_ETC_SLOPE | Other policy loan balance change | numeric | Trend |
| 26 | LN_AMT_YN | Policy loan holder | binary | 1=yes, 0=no |
| 27 | BIS_AREA_MOST | Most frequent lending sector | categorical | Both periods |
| 28 | BIS_AREA_MOST_RATIO | Ratio of most frequent lending sector | numeric | Both periods |
| 29 | BIS_AREA_NUM | Number of lending sectors | numeric | Both periods |
| 30 | BIS_AREA_FLAG_01 | Lending sector flag 01 | binary | 1=yes, 0=no |
| 31 | BIS_AREA_FLAG_02 | Lending sector flag 02 | binary | 1=yes, 0=no |
| 32 | BIS_AREA_FLAG_03 | Lending sector flag 03 | binary | 1=yes, 0=no |
| 33 | BIS_AREA_FLAG_04 | Lending sector flag 04 | binary | 1=yes, 0=no |
| 34 | BIS_AREA_FLAG_05 | Lending sector flag 05 | binary | 1=yes, 0=no |
| 35 | BIS_AREA_FLAG_06 | Lending sector flag 06 | binary | 1=yes, 0=no |
| 36 | BIS_AREA_LAST | Most recent lending sector | categorical | Latest |
| 37 | BIS_AREA_MAX | Largest lending sector (by contract amount) | categorical | Both periods |
| 38 | BIS_AREA_MAX_RATIO | Ratio of largest lending sector | numeric | Both periods |
| 39 | LN_GOODS_MOST | Most common loan product | categorical | Both periods |
| 40 | LN_GOODS_MOST_RATIO | Ratio of most common loan product | numeric | Both periods |
| 41 | LN_GOODS_NUM | Number of loan product types | numeric | Both periods |
| 42 | LN_GOODS_MAX | Largest loan product (by contract amount) | categorical | Both periods |
| 43 | LN_GOODS_MAX_RATIO | Ratio of largest loan product | numeric | Both periods |
| 44 | LN_CONT_MEAN | Average contracted loan amount | numeric | Both periods |
| 45 | LN_CONT_MAX | Maximum contracted loan amount | numeric | Both periods |
| 46 | TX_TP_MOST | Most common loan type | categorical | Both periods |
| 47 | TX_TP_MOST_RATIO | Ratio of most common loan type | numeric | Both periods |
| 48 | TX_TP_MAX | Largest loan type (by contract amount) | categorical | Both periods |
| 49 | TX_TP_MAX_RATIO | Ratio of largest loan type | numeric | Both periods |
| 50 | TX_TP_NUM | Number of loan types | numeric | Both periods |
| 51 | TX_TP_FLAG_01 | Loan type flag 01 | binary | 1=yes, 0=no |
| 52 | TX_TP_FLAG_02 | Loan type flag 02 | binary | 1=yes, 0=no |
| 53 | TX_TP_FLAG_03 | Loan type flag 03 | binary | 1=yes, 0=no |
| 54 | TX_TP_FLAG_04 | Loan type flag 04 | binary | 1=yes, 0=no |
| 55 | TX_TP_FIRST | Earliest loan type | categorical | Initial period |
| 56 | TX_TP_LAST | Most recent loan type | categorical | Latest |
| 57 | FND_PURP_MAX | Largest loan purpose (by contract amount) | categorical | Both periods |
| 58 | FND_PURP_MAX_RATIO | Ratio of largest loan purpose | numeric | Both periods |
| 59 | FND_PURP_NUM | Number of loan purposes | numeric | Both periods |
| 60 | FND_PURP_FLAG_01 | Loan purpose flag 01 | binary | 1=yes, 0=no |
| 61 | FND_PURP_FLAG_02 | Loan purpose flag 02 | binary | 1=yes, 0=no |
| 62 | FND_PURP_FLAG_03 | Loan purpose flag 03 | binary | 1=yes, 0=no |
| 63 | FND_PURP_FLAG_04 | Loan purpose flag 04 | binary | 1=yes, 0=no |
| 64 | OPN_BS_COUNT | Newly opened loan accounts | numeric | Compared to previous year |
| 65 | MRTY_BS_COUNT | Matured loan accounts | numeric | Compared to previous year |
| 66 | LN_TERM_MAX | Maximum loan term | numeric | Both periods |
| 67 | LN_TERM_SRT_RATIO | Ratio of short-term loans (≤13 months) | numeric | Both periods |
| 68 | LN_TERM_LNG_RATIO | Ratio of long-term loans (≥62 months) | numeric | Both periods |
| 69 | LN_CONT_LN_LST_RATIO | Ratio of total balance to total contracted amount | numeric | Latest |
| 70 | LN_FIN_RATIO | Loan repayment completion ratio | numeric | Both periods |
| 71 | LN_MOST_RATIO | Balance concentration ratio | numeric | Both periods |
| 72 | LN_BAL_MEAN | Average outstanding loan balance | numeric | - |
| 73 | LN_BAL_SLOPE | Change in total loan balance | numeric | Both periods |
| 74 | DSR | Debt Service Ratio | numeric | Both periods |
| 75 | LN_COUNT | Number of loan accounts | numeric | Both periods |
| 76 | HOM_IN_SMA_YN | Residence in Seoul metropolitan area | binary | 1=yes, 0=no (latest) |
| **Target** | DLQ_ANY_YN | Delinquency occurrence | binary | 1=yes, 0=no |

---

## Experiment Codes

### Training and Reproducibility

- [Data Preprocess]()
- [Feature Engineering]()
- [Vanilla Model Experiment]()
- [Hyper-Parameter Tuning & Model Experiment]()
- [Two Stage Evaluation]()


### Model Screening
The repository provides baseline training and decoding scripts for:

**Machine Learning**
- Logistic Regression
- Random Forest
- XGBoost

**Deep Learning**
- 1D-CNN
- TabNet
- Transformer

**Evaluation design**
- stratified 5-fold cross-validation
- borrower-level fold handling
- numerical standardization and categorical encoding within folds
- ROC-AUC as primary metric
- PR-AUC, KS, precision, recall, F1, and confusion-matrix metrics as secondary metrics in the manuscript
- random seed 42
- Search design:
  - classical ML randomized search: 50 iterations
  - deep learning randomized search: 20 iterations with early stopping
 
---

### Intended Run Order

After editing `configs/paths.yaml` and filling local paths in a restricted environment, run commands from this audit folder root:

```bash
python src/DataPreprocess/data_preprocess.py --config configs/paths.yaml
python src/DataPreprocess/cohort_filtering.py --config configs/paths.yaml
python src/FeatureEngineering/feature_engineering.py --config configs/paths.yaml
python src/FeatureEngineering/feature_selection.py --config configs/paths.yaml
python src/Experiments/vanilla_model.py --config configs/paths.yaml
python src/Experiments/hyper_parameter_model.py --config configs/paths.yaml
python src/Experiments/two_stage_evaluation.py --config configs/paths.yaml
python src/Discussion/proxy_loss.py --config configs/paths.yaml
```

---

## Discussion
**Proxy Loss**

To examine the potential loss implications of FN recovery, an additional retrospective simulation was conducted using a simplified proxy-based loss formulation.
Proxy loss formula:

```
Proxy Loss = sum_i D_i * EAD_i * LGD
```

where:

- `D_i` is the observed delinquency indicator
- `EAD_i` is the available loan-exposure proxy
- `LGD` is a simplifying assumption

The Reproducibility Code is: [Proxy_Loss]()

---

## Experimental Setup

All experiments in this study were conducted under a consistent environment and evaluation criteria. The computational environment consisted of a 13th Gen Intel® Core™ i7-13700K CPU, and an NVIDIA GeForce RTX 4090 GPU with 24 GB of memory. Models were implemented using Python 3.10.20, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.7.2, and Tensorflow 2.19.0.
