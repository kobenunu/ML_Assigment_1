# ML Assignment 1 - Soft Split Decision Trees

Implementation of soft split decision trees with repeated K-fold cross-validation.

## Project Structure

```
.
├── Main.py                      # Data preprocessing
├── SoftSplitTreeModel.py        # Soft split decision tree classifier
├── TrainFlow.py                 # Model training utilities
├── KFoldCrossValidation.py      # Cross-validation implementation
├── run_cross_validation.py      # Cross-validation runner
├── requirements.txt             # Dependencies
├── datasets/                    # Raw data
├── datasets/processed/          # Preprocessed data
└── results/                     # Output files
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

This runs 5-fold cross-validation with 2 repeats. Results are saved to results/ directory including summary tables and sensitivity analysis.

## Implementation

The soft split decision tree modifies predict_proba to use probabilistic routing. At each split node there is an alpha probability of routing to the opposite direction. Each prediction is run n_runs times and averaged.

Default parameters: n_splits=5, n_repeats=2, alpha=0.1, n_runs=100
