import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# 1. Charger les données
df = pd.read_csv("data/raw/full_leish_data_2010-2022.csv")

# 2. NETTOYAGE CRUCIAL : Supprimer les espaces invisibles des noms de colonnes
df.columns = df.columns.str.strip()

# 3. Les 8 colonnes corrélées à supprimer impérativement
cols_to_drop = ['SOIL_W', 'RH2M', 'SOIL_M', 'WD10M', 'TMP_MIN', 'TMP_MAX', 'UVA', 'WS2M']

# 4. Préparer X et y
df['Cases_Class'] = pd.qcut(df['Cases'], q=2, labels=[0, 1])
X = df.drop(columns=['Cases', 'Cases_Class', 'Mois', 'Date'] + cols_to_drop, errors='ignore')
y = df['Cases_Class']

print(f"Colonnes conservées pour l'entraînement ({len(X.columns)}) : {list(X.columns)}")

# 5. Entraîner le pipeline sur les 9 colonnes propres
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

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

pipeline.fit(X_train, y_train)

# 6. Remplacer l'ancien modèle par le nouveau directement à la racine
joblib.dump(pipeline, 'leishmaniasis_pipeline.pkl')
print("Succès ! Modèle XGBoost enregistré à la racine : 'leishmaniasis_pipeline.pkl'")