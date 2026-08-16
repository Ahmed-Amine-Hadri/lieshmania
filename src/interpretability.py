import os
import shap
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def afficher_shap(donnees_tabulaires):
    st.subheader("Interprétation du Modèle Climatique (XGBoost) via SHAP")
    st.write("Ce graphique illustre comment chaque variable influence la probabilité prédite vers le haut (rouge) ou vers le bas (bleu).")
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chemin_pipeline = os.path.join(base_dir, 'leishmaniasis_pipeline.pkl')
        pipeline = joblib.load(chemin_pipeline)
        
        df_instance = pd.DataFrame([donnees_tabulaires])
        
        def predict_fn(X):
            df_temp = pd.DataFrame(X, columns=df_instance.columns)
            return pipeline.predict_proba(df_temp)[:, 1]
        
        # --- CORRECTION SHAP ---
        # On charge le vrai jeu de données pour servir de "référence" (background)
        csv_path = os.path.join(base_dir, 'data', 'raw', 'full_leish_data_2010-2022.csv')
        df_bg = pd.read_csv(csv_path)
        df_bg.columns = df_bg.columns.str.strip() # On nettoie les espaces
        
        cols_to_drop = ['SOIL_W', 'RH2M', 'SOIL_M', 'WD10M', 'TMP_MIN', 'TMP_MAX', 'UVA', 'WS2M', 'Cases', 'Cases_Class', 'Mois', 'Date']
        df_bg = df_bg.drop(columns=[c for c in cols_to_drop if c in df_bg.columns], errors='ignore')
        
        # On prend un échantillon de 50 lignes comme base de comparaison pour que le calcul soit très rapide
        background = shap.sample(df_bg, 50)
        
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(df_instance)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.summary_plot(shap_values, df_instance, plot_type="bar", show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'exécution de SHAP : {e}")