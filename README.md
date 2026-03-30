# PAQ Affect UI

A Flask-based web application for running a multi-stage psychology experiment studying facial expression perception and emotional affect. Participants complete pairwise comparison tasks in affect recognition before/after a perceptual adjustment query (PAQ) based calibration.

---

## Overview

The experiment follows a three-stage paradigm:

1. **Stage 1 — Pairwise-Early:** Participants compare pairs of facial images and indicate whether this pair of faces appears the same expression or not. Baseline response times (RT) are collected.
2. **Stage 2 — PAQ (Psychophysical Adjustment):** Participants use a slider to select the face that sits at their personal Just Noticeable Difference (JND) threshold.
3. **Stage 3 — Stage2-from-PAQ:** The pairwise task is repeated, now with the right-side image replaced by each participant's individually chosen JND value. Post-PAQ RTs are collected.

---

## Project Structure

```
paq_affect_UI/
├── app.py                        # Main Flask web application (baseline vs PAQ)
├── app-1.py                      # Main Flask Web application (Weibull vs PAQ)
├── data_analysis.py              # Lighweight script of RT analysis and per-user visualizations
├── rt_analysis.py                # Full statistical analysis (RT vs stimi, per-user summary stats, distribution 
|                                 # plots, group descriptives, ANOVA, variance decomposition, ex-Gaussian, Wilcoxon test)
├── build_pairs_multi.py          # one time setup script as config generator: scans images, builds endpoints.json
├── static/
│   └── images/
│       ├── <category>/           # Facial image sequences (001.png – 100.png)
│       │   ├── ref_face/         # Reference images
│       │   └── compress_img.py   # Image compression utility (run per-category)
│       └── *.png                 # Stage preview images for introduction of stage 2 pairwise comparison task
├── config/
│   ├── endpoints.json            # Generated image endpoint mappings
│   ├── endpoints_log.json        # Validation log from build_pairs_multi.py
│   ├── endpoints_flat.csv        # Flattened endpoint list
│   └── pairs.json                # Explicit pair definitions
├── submissions/
│   └── pairwise/                 # Per-session JSON data files (generated at runtime)
├── figures/                      # Analysis output figures and CSVs
├── psychometric_results.csv      # Psychometric calibration data (JND, floor_int per pair)
├── requirements.txt
└── README.md
```

### Face Identity Categories

Images are organized into 8 identity categories (combinations of gender and apparent ethnicity):

| Code | Description           |
|------|-----------------------|
| `bm` | Black Male            |
| `bf` | Black Female          |
| `wm` | White Male            |
| `wf` | White Female          |
| `mm` | Malaysian Male        |
| `mf` | Malaysian Female      |
| `im` | Indian Male           |
| `if` | Indian Female         |

Each category has a `_sad` and `_happy` variant, containing 100 morphed frames from a neutral baseline toward the target expression.

---

## Python Files

### `app.py`
The main Flask server. Handles all experiment routing, session management, trial order construction, and data persistence.

**Routes summary:**

| Route | Purpose |
|-------|---------|
| `/` | Landing page |
| `/introduction` | Study intro |
| `/pretest` | Pre-test questionnaire |
| `/pairwise/intro` → `/pairwise/trial` | Stage 1 trials |
| `/pairwise/ask_difficulty` | Per-block difficulty rating |
| `/pairwise/done` | Stage 1 completion |
| `/paq/index` → `/paq/<folder>/<ref_idx>/` | PAQ slider task |
| `/paq/api/submit` | PAQ data submission endpoint |
| `/stage2/intro` → `/stage2/trial` | Stage 3 trials |
| `/stage2/ask_difficulty` | Per-block difficulty rating |
| `/stage2/api/submit` | Stage 3 data submission |
| `/end` | Final summary page |

---

### `app-1.py`
Another version of `app.py` with Weibull-calibrated JND as stage 1 facial pairwise comparison task. Retained for reference. `app.py` is the active application.

---

### `data_analysis.py`
Basic post-experiment analysis script (~350 lines).

**What it does:**
- Loads participant JSON submissions from `submissions/pairwise/`
- Builds RT matrices for before-PAQ and after-PAQ blocks
- Plots per-user RT curves showing Before vs. After comparison
- Runs per-user linear regression (slope of RT vs. stimulus index)
- Performs one-sample t-tests on slope distributions (Before, After, and difference)

**Outputs:** PNG figures saved to `figures/`

---

### `rt_analysis.py`
Advanced behavioral analysis script (~810 lines) for publication-quality statistics.

**Analyses included:**
- **Descriptive stats:** Mean, SD, and skewness per participant × condition
- **KDE plots:** Per-participant RT distributions (before vs. after)
- **Log-normal fitting:** Group-level RT distribution modeling
- **Paired t-tests:** On mean RT, SD, and skewness between conditions
- **Repeated-measures ANOVA:** 2-way (Condition × Expression) using `pingouin`
- **Variance decomposition:** Between-subject vs. within-subject variance
- **Ex-Gaussian fitting:** Estimates μ, σ, τ parameters (drift diffusion model basis)
- **Per-user CSV export:** Mean and variance RT by condition to `figures/user_summary_before_after.csv`

**Outputs:** 9+ figures saved to subdirectories within `figures/`

---

### `build_pairs_multi.py`
Configuration generator (~130 lines). Run once to set up image endpoint mappings before serving the experiment.

**What it does:**
- Scans `static/images/{category}/ref_face/` for reference endpoint images
- Validates that both `lo` and `hi` endpoints exist for each index (1–100)
- Groups images into intervals of 10 (e.g., [1–10], [11–20], …)
- Writes results to `config/endpoints.json` and `config/endpoints_log.json`

---

### `static/images/<category>/compress_img.py`
Small utility script (~24 lines) for preprocessing facial images before deployment.

**What it does:**
- Converts images to RGB (strips alpha channels)
- Resizes to max 1024×1024
- Optimizes PNG compression
- Outputs compressed files to `./images_compressed/`

Run from within a specific category folder as needed.

---

## Data Files

### `psychometric_results.csv`
Psychometric calibration data with one row per reference face pair. Key columns:

| Column | Description |
|--------|-------------|
| `pair_id` | Identity pair identifier |
| `jnd` | Just Noticeable Difference threshold |
| `floor_int` | Integer offset applied to pairwise trial right-image index |

Used by `app.py` to calibrate trial difficulty at the group level.

### `submissions/pairwise/{session_id}.json`
Generated at runtime. One file per participant, containing all trial-level data:
- Stimulus identities, image indices, response choices
- Response times (milliseconds, captured client-side)
- Block structure, difficulty ratings
- PAQ-selected frame numbers

---

## Setup and Installation

### Dependencies

Install Flask and analysis libraries:

```bash
pip install flask pandas numpy scipy matplotlib seaborn pingouin pillow
```

> **Note:** `requirements.txt` lists FastAPI and Uvicorn, which are not used by the current application. The active server framework is Flask.

---

## Running the Application

### 1. Generate image configuration (first-time setup)

```bash
python build_pairs_multi.py
```

This scans the image directory and writes `config/endpoints.json`.

### 2. Start the Flask development server

```bash
python app.py
```

Open a browser and navigate to the local url.


---

## Running Data Analysis

After collecting participant data in `submissions/pairwise/`, run one of the analysis scripts:

```bash
# Basic RT analysis and per-user plots
python data_analysis.py

# Full statistical analysis (ANOVA, ex-Gaussian, variance, etc.)
python rt_analysis.py
```

Output figures are saved to the `figures/` directory.

---

## Experiment Flow

```
Participant arrives
      │
      ▼
  / (landing page)
      │
      ▼
  /introduction
      │
      ▼
  /pretest
      │
      ▼
  Stage 1: Pairwise-Early
  /pairwise/intro → /pairwise/trial (×N) → /pairwise/ask_difficulty → /pairwise/done
      │
      ▼
  Stage 2: PAQ
  /paq/index → /paq/<folder>/<ref_idx>/ (slider) → /paq/api/submit
      │
      ▼
  Stage 3: Stage2-from-PAQ
  /stage2/intro → /stage2/trial (×N) → /stage2/ask_difficulty
      │
      ▼
  /end (summary + Prolific redirect)
```

---

## Technical Notes

- **Counterbalancing:** Participants are randomly assigned to Group A or Group B, each covering 4 of the 8 face identities.
- **Client-side timing:** Response times are captured in JavaScript (milliseconds) and submitted asynchronously.
- **Atomic writes:** Session JSON files are written via a temporary file + atomic rename to prevent corruption if the server is interrupted.

