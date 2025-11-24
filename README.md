
# UK Riverbank Water Quality Classification

This project analyzes and classifies water quality in UK riverbanks using machine learning. The workflow includes data cleaning, feature engineering, model training, and evaluation.

## Data
- Currently not available publicly.
- Target variable: `phosphate` (classified into discrete classes)
- Features: Geographic, temporal, physical, and categorical attributes

## Workflow
1. **Load & Clean Data**: Remove missing and extreme outlier values for the target.
2. **Feature Engineering**: Create time-based, interaction, and categorical features.
3. **Train/Test Split**: Stratified split for robust evaluation.
4. **Preprocessing**: Standard scaling for numerics, one-hot encoding for categoricals, boolean passthrough.
5. **Model Training**: Random Forest and Logistic Regression with cross-validation.
6. **Evaluation**: Accuracy, F1 score, classification report, per-class metrics, and feature importance.
7. **Baselines**: Dummy classifiers for random and most frequent class comparison.

## Usage
Run the main training script:

```bash
python train.py
```

Outputs include:
- Model performance metrics
- Feature importance plot (`feature_importance.png`)
- Per-class performance table

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, matplotlib

## Results
The script compares multiple models and baselines, reporting cross-validation and test metrics. Feature importances are visualized for the best model.
