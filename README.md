# Algorithmic fairness and clinical utility of AI-based knee osteoarthritis grading across demographic subgroups

This repository contains the analysis and model interfaces for a multi-cohort audit of five-class Kellgren–Lawrence grading. It evaluates discrimination, calibration, ordinal error direction, demographic disparity, intersectional effects, temporal stability, and clinical utility at the KL-1/KL-2 treatment boundary.

## Scope

The implementation covers five model families: a medial/lateral Siamese CNN, VGG-19 with ordinal loss, DenseNet, a probability ensemble, and a Swin–EfficientNet interaction model. The audit supports OAI baseline and longitudinal visits and a second cohort with the same manifest schema.

The ten formal quantities in the study are implemented directly:

- Per-grade equalized odds ratio
- Per-grade equalized odds difference
- Per-grade demographic parity difference
- Calibration difference using ten equal-width bins
- Worst-group macro-AUC gap
- Per-grade false-negative-rate disparity
- Clinically weighted Ordinal Disparity Index
- Under-grading rate
- Under-grading disparity
- Intersectional compounding penalty

The statistical layer provides 1,000-resample stratified or participant-clustered bootstrap intervals, 10,000-label permutation tests, Bonferroni–Holm family-wise correction, Benjamini–Hochberg false-discovery correction, Cohen's d, Kendall concordance, Spearman correlation, and Friedman temporal tests.

## Installation

Python 3.10 is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda installation:

```bash
conda env create -f environment.yml
conda activate knee-fairness-audit
pip install --no-deps -e .
```

Container installation:

```bash
docker build -t knee-fairness-audit .
```

## Data

The verified official OAI access page is listed in `dataset_links.txt`. Access requires repository registration and acceptance of its terms. The source images are not redistributed.

Prepare a CSV manifest with these columns:

```text
image_path,participant_id,knee,visit,grade,race,sex,age,bmi,cohort
```

Each row represents one knee at one visit. `knee` is `left` or `right`; `grade` is an integer from 0 through 4. Use stable study identifiers in `participant_id`, never names, dates of birth, medical-record numbers, addresses, or other direct identifiers. Relative image paths are resolved from the manifest directory.

The primary OAI analysis uses V00. Temporal evaluation uses V00, V01, V03, V06, and V10. Split assignment is participant-level so images from the same participant cannot cross training, validation, and test partitions. The external cohort must be kept separate from all OAI-trained model fitting and calibration.

## Training

```bash
knee-fairness-train \
  --config configs/main.yaml \
  --manifest data/oai_manifest.csv \
  --output outputs/densenet.pt
```

The paper does not state batch size, learning rate, epoch count, optimizer, scheduler, precision, GPU model, GPU count, wall-clock time, or storage use. Values in `configs/main.yaml` are transparent engineering defaults and are marked with `paper_training_parameters_reported: false`; they are not asserted as reported study parameters. Hardware and duration therefore depend on the selected architecture, image resolution, and local system.

Each saved state contains model, optimizer, scheduler, epoch, global step, seed, best validation metric, CPU random state, and CUDA random states. Writes use a temporary file followed by an atomic replacement.

## Prediction table

The evaluation command consumes a CSV containing:

```text
participant_id,label,group,p0,p1,p2,p3,p4
```

Probabilities must be non-negative and sum to one within numerical tolerance. `group` contains the selected demographic axis. Prediction tables should contain study identifiers only.

## Evaluation

Race audit:

```bash
knee-fairness-evaluate \
  --predictions outputs/oai_race_predictions.csv \
  --left-group White \
  --right-group "African American" \
  --output outputs/oai_race_audit.json
```

Repeat the command for sex, age, and BMI after selecting the corresponding group column. Report only cells with at least 30 observations. Primary comparisons use Holm correction across the five models and four axes. Per-grade and intersectional analyses use false-discovery-rate correction at q = 0.05.

Expected OAI aggregate values are stored in `knee_fairness.reference` for validation of completed runs. Multiclass accuracy spans 66.4% to 77.2%, weighted kappa spans 0.82 to 0.87, and macro-AUC spans 0.84 to 0.89. Racial macro-AUC gaps span 3.1 to 6.8 percentage points. The acceptance tolerance for architecture-level accuracy is two percentage points.

## Clinical utility

Clinical utility uses cumulative probability for KL grade 2 or higher. Temperature scaling is fit on a held-out calibration partition disjoint from training and testing. Decision curves compare model net benefit with treat-all and treat-none strategies over threshold probabilities from 0.01 through 0.99. Net reclassification improvement is calculated separately for events and non-events and then stratified by group.

Group-aware thresholds are fitted with the within-group Youden index at the KL-1/KL-2 boundary. This is an analysis of post-hoc mitigation, not a substitute for reporting unadjusted subgroup performance. Both the original and recalibrated results should be retained in an audit export.

## Output interpretation

Equalized odds ratio values of at least 0.80 indicate approximate parity. Equalized odds difference below 0.10, demographic parity difference at most 0.10, calibration difference at most 0.05, worst-group gap at most five percentage points, and false-negative-rate disparity at most 0.05 are the study thresholds.

Race is treated as a social stratifier representing structural, environmental, and healthcare factors. It must not be interpreted as a biological cause. Subgroup results require confidence intervals and cell counts. Comparisons below the minimum cell size are excluded from inference.

## Repository layout

```text
code/knee_fairness
  audit.py
  calibration.py
  clinical.py
  config.py
  data
  evaluation.py
  fairness.py
  losses.py
  metrics.py
  models
  reference.py
  reporting.py
  schema.py
  statistics.py
  training
configs
dataset_links.txt
```

No clinical decision should be made from a model output without local validation, qualified human review, and a documented escalation pathway for discordant symptoms and radiographic grades.

