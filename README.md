# ML Assignment 1 - Soft Split Decision Trees

Implementation of soft split decision trees with repeated K-fold cross-validation and sensitivity analysis.

## Project Structure

```
.
├── Main.py                      # Data preprocessing
├── SoftSplitTreeModel.py        # Soft split decision tree classifier
├── TrainFlow.py                 # Model training utilities
├── KFoldCrossValidation.py      # Cross-validation implementation
├── SensitivityAnalysis.py       # Sensitivity analysis module
├── run_cross_validation.py      # Cross-validation runner
├── run_sensitivity_analysis.py  # Sensitivity analysis runner
├── GenerateVisualizations.py    # Visualization generator
├── requirements.txt             # Dependencies
├── datasets/                    # Raw data
├── datasets/processed/          # Preprocessed data
├── results/                     # Cross-validation results
├── results/sensitivity/         # Sensitivity analysis results
└── results/visualizations/      # Generated plots
```

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preprocessing

Preprocess raw datasets:

```bash
python Main.py
```

Processed files are saved to datasets/processed/ with the target column renamed to 'target'.

## K-Fold Cross-Validation

Run cross-validation on all datasets:

```bash
python run_cross_validation.py
```

This runs 5-fold cross-validation with 2 repeats. Results are saved to results/ directory including summary tables and detailed fold results.

## Sensitivity Analysis

Test different alpha and n_runs parameter values:

```bash
python run_sensitivity_analysis.py
```

This evaluates performance across:

-   Alpha values: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
-   n_runs values: [10, 25, 50, 75, 100, 150, 200]

Results are saved to results/sensitivity/ directory.

## Generate Visualizations

Create plots from sensitivity analysis results:

```bash
python GenerateVisualizations.py
```

Generates:

-   Alpha sensitivity plots (performance vs alpha)
-   n_runs sensitivity plots (performance vs n_runs)
-   Heatmaps showing parameter combinations
-   Cross-dataset comparison plots
-   Best parameters summary

Plots are saved to results/visualizations/ directory.

## Implementation

The soft split decision tree modifies predict_proba to use probabilistic routing. At each split node there is an alpha probability of routing to the opposite direction. Each prediction is run n_runs times and averaged.

Default parameters: n_splits=5, n_repeats=2, alpha=0.1, n_runs=100
