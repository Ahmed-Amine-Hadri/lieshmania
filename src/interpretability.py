import shap
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def afficher_shap(donnees_tabulaires):
    st.subheader("Interprétation du Modèle Climatique (XGBoost) via SHAP")
    st.write("Ce graphique illustre comment chaque variable influence la probabilité prédite vers le haut (rouge) ou vers le bas (bleu).")
    
    try:
        # Chemin pointant vers le dossier notebooks/
        pipeline = joblib.load('notebooks/leishmaniasis_pipeline.pkl')
        df_instance = pd.DataFrame([donnees_tabulaires])
        
        def predict_fn(X):
            df_temp = pd.DataFrame(X, columns=df_instance.columns)
            return pipeline.predict_proba(df_temp)[:, 1]
        
        background = pd.DataFrame([donnees_tabulaires]) 
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(df_instance)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.summary_plot(shap_values, df_instance, plot_type="bar", show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'exécution de SHAP : {e}")