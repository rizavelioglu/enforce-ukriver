
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

Here's an example output:
```bash
=== Training RandomForest ===
CV F1 Score: 0.5674 (+/- 0.0087)
Test Accuracy: 0.6160
Test F1 Score: 0.5928
              precision    recall  f1-score   support

         0.0      0.713     0.866     0.782      2134
       100.0      0.503     0.496     0.499      1223
       200.0      0.445     0.283     0.346       573
       300.0      0.427     0.249     0.315       281
       500.0      0.392     0.199     0.264       146
      1000.0      0.477     0.304     0.372        69
      2500.0      0.667     0.182     0.286        22

    accuracy                          0.616      4448
   macro avg      0.518     0.368     0.409      4448
weighted avg      0.588     0.616     0.593      4448

=== Training LogReg ===
CV F1 Score: 0.5716 (+/- 0.0055)
Test Accuracy: 0.6005
Test F1 Score: 0.5838
              precision    recall  f1-score   support

         0.0      0.748     0.826     0.785      2134
       100.0      0.486     0.531     0.508      1223
       200.0      0.365     0.283     0.319       573
       300.0      0.332     0.238     0.277       281
       500.0      0.246     0.116     0.158       146
      1000.0      0.333     0.145     0.202        69
      2500.0      0.250     0.091     0.133        22

    accuracy                          0.600      4448
   macro avg      0.394     0.319     0.340      4448
weighted avg      0.575     0.600     0.584      4448

=== RANDOM CLASSIFIER (uniform) ===
Accuracy: 0.1432 (vs 0.1429 expected)
Classification Report:
              precision    recall  f1-score   support

           0       0.47      0.14      0.22      2134
           1       0.28      0.14      0.19      1223
           2       0.14      0.15      0.14       573
           3       0.05      0.13      0.08       281
           4       0.03      0.16      0.06       146
           5       0.02      0.16      0.03        69
           6       0.01      0.23      0.02        22

    accuracy                           0.14      4448
   macro avg       0.14      0.16      0.10      4448
weighted avg       0.32      0.14      0.18      4448

=== STRATIFIED RANDOM CLASSIFIER ===
Accuracy: 0.3336
Classification Report:
              precision    recall  f1-score   support

           0       0.48      0.47      0.48      2134
           1       0.29      0.29      0.29      1223
           2       0.14      0.14      0.14       573
           3       0.11      0.10      0.10       281
           4       0.02      0.02      0.02       146
           5       0.02      0.01      0.02        69
           6       0.00      0.00      0.00        22

    accuracy                           0.33      4448
   macro avg       0.15      0.15      0.15      4448
weighted avg       0.33      0.33      0.33      4448


=== MOST FREQUENT CLASS CLASSIFIER ===
Accuracy: 0.4798
Most common class in train: 0, appears in test: 47.9766%
```

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, matplotlib

## Results
The script compares multiple models and baselines, reporting cross-validation and test metrics. Feature importances are visualized for the best model.
