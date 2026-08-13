import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score
from sklearn.dummy import DummyClassifier
import joblib
import os

# --- 0. CHARGEMENT CORRECT DES DONNÉES ---
csv_path = os.path.join("data", "raw", "full_leish_data_2010-2022.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[OK] Données chargées depuis {csv_path} (Lignes: {df.shape[0]}, Colonnes: {df.shape[1]})")
else:
    raise FileNotFoundError(f"Impossible de trouver le fichier CSV à l'emplacement : {csv_path}")

# Sécurité pour éviter les erreurs si 'cols_to_drop' n'est pas définie
cols_to_drop = []

# 1. Define Target and Features FIRST
df['Cases_Class'] = pd.qcut(df['Cases'], q=2, labels=[0, 1])
X = df.drop(columns=['Cases', 'Cases_Class', 'Mois', 'Date'] + cols_to_drop, errors='ignore')
y = df['Cases_Class']

# 2. Split the data before ANY transformations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# 3. Calculate Baseline Error (Avoidable Bias reference)
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
baseline_acc = dummy_clf.score(X_train, y_train)
baseline_error = 1 - baseline_acc

# 4. Build the Leak-Free Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('xgb', xgb.XGBClassifier(
        max_depth=2,
        learning_rate=0.01,
        n_estimators=100,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=2.0,
        reg_lambda=15.0,
        eval_metric='logloss',
        random_state=42
    ))
])

# 5. Cross-Validation (The Pipeline prevents leakage inside the folds)
cv_results = cross_validate(
    pipeline, 
    X_train, 
    y_train, 
    cv=5, 
    scoring=['accuracy', 'precision'], 
    return_train_score=True
)

# Extract Mean Metrics from CV
train_accuracy = np.mean(cv_results['train_accuracy'])
cv_accuracy = np.mean(cv_results['test_accuracy'])
train_precision = np.mean(cv_results['train_precision'])
cv_precision = np.mean(cv_results['test_precision'])

# 6. Final Evaluation on Holdout Set
pipeline.fit(X_train, y_train)
test_predictions = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, zero_division=0)

# 7. Bias / Variance Calculations
train_error = 1 - train_accuracy
cv_error = 1 - cv_accuracy

bias = train_error
avoidable_bias = max(0, train_error - baseline_error)
variance = max(0, cv_error - train_error)

# ==========================================
# PRINT RESULTS & DIAGNOSTICS
# ==========================================

print("--- CLASSIFICATION METRICS ---")
print(f"Accuracy (Train) : {train_accuracy * 100:.2f}%")
print(f"Accuracy (CV)    : {cv_accuracy * 100:.2f}%")
print(f"Accuracy (Test)  : {test_accuracy * 100:.2f}%\n")

print(f"Precision (Train): {train_precision * 100:.2f}%")
print(f"Precision (CV)   : {cv_precision * 100:.2f}%")
print(f"Precision (Test) : {test_precision * 100:.2f}%\n")

print("--- DIAGNOSTICS BIAIS / VARIANCE ---")
print(f"Erreur de Base (Baseline) : {baseline_error:.4f} (Prédiction de la classe majoritaire)")
print(f"Biais Total               : {bias:.4f}")
print(f"Biais Évitable            : {avoidable_bias:.4f}")
print(f"Variance                  : {variance:.4f}\n")

if avoidable_bias > variance:
    print("Diagnostic : SOUS-APPRENTISSAGE (Underfitting).")
elif variance > avoidable_bias:
    print("Diagnostic : SUR-APPRENTISSAGE (Overfitting).")
else:
    print("Diagnostic : Équilibre stable atteint.")

# ==========================================
# SAVE RESULTS & PIPELINE
# ==========================================

results_df = X_test.copy()
results_df['Actual_Class'] = y_test
results_df['Predicted_Class'] = test_predictions
results_df.to_csv('xgboost_test_predictions.csv', index=False)
print("\nSaved predictions to 'xgboost_test_predictions.csv'")

joblib.dump(pipeline, 'leishmaniasis_pipeline.pkl')
print("Saved the trained pipeline to 'leishmaniasis_pipeline.pkl'")