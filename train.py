import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier

# === 1. Load & Clean Data ===
df = pd.read_csv("<replace-with-location-of-data>", index_col=0)
df = df.iloc[:, 1:]  # Drop ID column

target = 'phosphate'
df_clean = df.dropna(subset=[target]).copy()
df_clean = df_clean[df_clean[target] != 150.00]  # Remove extreme outlier

# Encode target classes
sorted_unique = sorted(df_clean[target].unique())
df_clean['phosphate_class'] = df_clean[target].map({val: i for i, val in enumerate(sorted_unique)})

# === 2. Feature Engineering ===
df_clean['datetime'] = pd.to_datetime(df_clean['timestamp'], unit='ms')
df_clean['month'] = df_clean['datetime'].dt.month
df_clean['season'] = df_clean['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
df_clean['year'] = df_clean['datetime'].dt.year
df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
df_clean['hour'] = df_clean['datetime'].dt.hour

# Add interaction features
df_clean['temp_turbidity'] = df_clean['temperature'] * df_clean['turbidity']
df_clean['tds_turbidity'] = df_clean['totalDissolvedSolids'] * df_clean['turbidity']
 
categorical_features = ['river', 'waterBodyType', 'waterFlow', 'waterLevel', 'MNCAT_NAME', 'OPCAT_NAME', 'site_cluster_label']
numerical_features = [
    'long', 'lat', 'estimatedWidth', 'estimatedDepth', 'temperature',
    'totalDissolvedSolids', 'turbidity', 'month', 'season', 'year', 
    'day_of_week', 'hour', 'temp_turbidity', 'tds_turbidity'
]
bool_features = df_clean.select_dtypes(include=['bool']).columns.tolist()

# Fill missing numeric values BEFORE splitting
# df_clean[numerical_features] = df_clean[numerical_features].fillna(
#     df_clean[numerical_features].median()
# )
df_clean = df_clean.dropna(subset=numerical_features)

# Create feature dataframe
df_features = df_clean[categorical_features + numerical_features + bool_features].copy()

# Drop extremely low variance bool features
low_variance_cols = [c for c in bool_features if df_features[c].mean() < 0.01]
df_features.drop(columns=low_variance_cols, inplace=True)
bool_features = [c for c in bool_features if c not in low_variance_cols]

# Convert booleans to 0/1
for col in bool_features:
    df_features[col] = df_features[col].astype(int)

# Convert categorical to string type
for col in categorical_features:
    df_features[col] = df_features[col].astype(str)

print(f"Final feature shape: {df_features.shape}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Boolean features: {len(bool_features)}")

# === 3. Train/Test Split ===
y = df_clean['phosphate_class'].values

X_train, X_test, y_train, y_test = train_test_split(
    df_features, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Preprocessing Pipeline with One-Hot Encoding ===
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bool', 'passthrough', bool_features)
    ]
)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed feature shape: {X_train_processed.shape}")

# === 5. Train Multiple Models with Cross-Validation ===
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    # "HistGB": HistGradientBoostingClassifier(),
    # "GB": GradientBoostingClassifier(random_state=42),
    "LogReg": LogisticRegression(random_state=42)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full training set
    model.fit(X_train_processed, y_train)
    
    # Predictions
    pred = model.predict(X_test_processed)
    
    # Metrics
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(classification_report(y_test, pred, digits=3, target_names=[str(c) for c in sorted_unique]))
    
    results.append({
        "Model": name,
        "CV_F1": cv_scores.mean(),
        "Test_Accuracy": acc,
        "Test_F1": f1
    })

# === 6. Results Comparison ===
print("\n=== Model Comparison ===")
results_df = pd.DataFrame(results).sort_values("Test_F1", ascending=False)
print(results_df)

# === 7. Feature Importance (Best Model) ===
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

if hasattr(best_model, "feature_importances_"):
    # Get feature names after one-hot encoding
    feature_names = []
    
    # Numerical features
    feature_names.extend(numerical_features)
    
    # One-hot encoded categorical features
    if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
    
    # Boolean features
    feature_names.extend(bool_features)
    
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(best_model.feature_importances_)],
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Plot top 20
    top20 = importance_df.head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top20)), top20["importance"], color='steelblue')
    plt.yticks(range(len(top20)), top20["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top 20 Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

# === 8. Confusion Matrix ===
pred_best = best_model.predict(X_test_processed)
# cm = confusion_matrix(y_test, pred_best)

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=sorted_unique, yticklabels=sorted_unique)
# plt.title(f"Confusion Matrix ({best_model_name})")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

# === 9. Per-Class Performance ===
from sklearn.metrics import classification_report
report = classification_report(y_test, pred_best, target_names=[str(c) for c in sorted_unique], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\n=== Per-Class Performance ===")
print(report_df)

# === 10. Baseline Comparisons with Dummy Classifiers ===
# Random classifier (uniform random guessing)
random_clf = DummyClassifier(strategy='uniform', random_state=42)
random_clf.fit(X_train, y_train)
y_pred_random = random_clf.predict(X_test)
random_accuracy = accuracy_score(y_test, y_pred_random)
print(f"\n=== RANDOM CLASSIFIER (uniform) ===")
print(f"Accuracy: {random_accuracy:.4f} (vs {1/(len(sorted_unique)):.4f} expected)")
print("Classification Report:")
print(classification_report(y_test, y_pred_random, zero_division=0))

stratified_clf = DummyClassifier(strategy='stratified', random_state=42)
stratified_clf.fit(X_train, y_train)
y_pred_strat = stratified_clf.predict(X_test)
stratified_accuracy = accuracy_score(y_test, y_pred_strat)
print(f"\n=== STRATIFIED RANDOM CLASSIFIER ===")
print(f"Accuracy: {stratified_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_strat, zero_division=0))

mf_clf = DummyClassifier(strategy='most_frequent', random_state=42)
mf_clf.fit(X_train, y_train)
y_pred_mf = mf_clf.predict(X_test)
mf_accuracy = accuracy_score(y_test, y_pred_mf)
most_common_class = np.bincount(y_train).argmax()
support_most_common = np.sum(y_test == most_common_class) / len(y_test)
print(f"\n=== MOST FREQUENT CLASS CLASSIFIER ===")
print(f"Accuracy: {mf_accuracy:.4f}")
print(f"Most common class in train: {most_common_class}, appears in test: {support_most_common:.4%}")
