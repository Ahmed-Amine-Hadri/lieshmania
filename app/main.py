import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import os
import torch
import joblib
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import shap

# --- CONFIGURATION ---
st.set_page_config(page_title="Leishmania-AI Diagnostic", layout="wide", page_icon="🔬")

# ==========================================
# CHARGEMENT DES MODÈLES (AVEC MISE EN CACHE)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = "timm-mobilenetv3_small_100"
PATCH_SIZE = 128

@st.cache_resource
def charger_modeles():
    # 1. Pipeline XGBoost
    xgb_path = 'leishmaniasis_pipeline.pkl'
    xgb_pipeline = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    
    # 2. Modèle U-Net CNN
    cnn_model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)
    
    weights_path = 'weights/unet_best.pth'
    if os.path.exists(weights_path):
        cnn_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        cnn_model.eval()
    
    # 3. Données pour SHAP (Explainer)
    csv_path = os.path.join("data", "raw", "full_leish_data_2010-2022.csv")
    df_shap = None
    if os.path.exists(csv_path) and xgb_pipeline is not None:
        df_shap = pd.read_csv(csv_path).head(100) # Échantillon pour calcul rapide SHAP
        
    return xgb_pipeline, cnn_model, df_shap

xgb_pipeline, cnn_model, df_shap = charger_modeles()

# Transformations d'images pour le CNN
transformations_inference = A.Compose([
    A.Resize(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- FONCTIONS DE PRÉDICTION ---
def predire_avec_cnn(image_pil):
    """Traite l'image PIL via le CNN U-Net."""
    image_np = np.array(image_pil.convert('RGB'))
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Pour compatibilité OpenCV si besoin
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    
    tenseur_image = transformations_inference(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = cnn_model(tenseur_image)
        probs_pixels = torch.sigmoid(logits)
        prob_lesion = probs_pixels.max().item()
        
    return np.array([1.0 - prob_lesion, prob_lesion])

def predire_avec_xgboost(donnees_tabulaires):
    """Traite les données tabulaires via XGBoost."""
    if xgb_pipeline is None:
        return np.array([0.5, 0.5])
    df_input = pd.DataFrame([donnees_tabulaires]) if isinstance(donnees_tabulaires, dict) else donnees_tabulaires
    return xgb_pipeline.predict_proba(df_input)[0]

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
st.sidebar.info("Outil de Détection & Analyse Intégré (CNN + XGBoost + SHAP)")
volet = st.sidebar.radio(
    "Select a module:", 
    ["Volet 1: Context & Statistics", "Volet 2: AI Diagnosis & Hybrid Fusion", "Volet 3: Model Explainability (SHAP)"]
)

# ==========================================
# VOLET 1: Context & Statistics
# ==========================================
if volet == "Volet 1: Context & Statistics":
    st.title("🌍 Leishmaniasis: Overview & Regional Impact")
    
    st.markdown("### What is Leishmaniasis?")
    st.write("""
    Leishmaniasis is a parasitic disease caused by the *Leishmania* parasite, transmitted through the bites of infected female phlebotomine sandflies. 
    The most common form is **Cutaneous Leishmaniasis (CL)**, which causes skin lesions, ulcers, and permanent scars.
    """)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📍 Focus: The Ouarzazate Region")
        st.write("""
        Ouarzazate is one of Morocco's primary endemic zones for Cutaneous Leishmaniasis due to environmental and ecological vectors.
        """)
        st.info("""
        **Key Statistics:**
        * **Predominant Strain:** *Leishmania major* and *Leishmania tropica*.
        * **Incidence Spikes:** Outbreaks follow rainy seasons.
        """)
        
    with col2:
        st.markdown("### Seasonal Case Distribution")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        cases = [120, 90, 60, 40, 30, 80, 150, 210, 340, 410, 280, 180]
        chart_data = pd.DataFrame({'Cases': cases}, index=months)
        st.bar_chart(chart_data, color="#ff4b4b")

# ==========================================
# VOLET 2: AI Diagnosis (Image/CNN + Tabular/XGB)
# ==========================================
elif volet == "Volet 2: AI Diagnosis":
    st.title("🔬 Clinical Hybrid Diagnosis (CNN + XGBoost)")
    st.write("Uploadez une image de lésion et/ou entrez les paramètres environnementaux pour une prédiction combinée.")
    
    col_img, col_data = st.columns([1, 1])
    
    with col_img:
        uploaded_file = st.file_uploader("Upload Lesion Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Clinical Image", use_container_width=True)
            
    with col_data:
        st.markdown("### Paramètres Tabulaires (Optionnel)")
        use_tabular = st.checkbox("Ajouter des données tabulaires (météo / environnement)")
        
        tab_data = {}
        if use_tabular:
            col_a, col_b = st.columns(2)
            with col_a:
                tab_data['Annee'] = st.number_input("Année", value=2026)
                tab_data['WS10M'] = st.number_input("Vent (WS10M)", value=2.5)
                tab_data['WD2M'] = st.number_input("Direction Vent (WD2M)", value=120.0)
                tab_data['UVB'] = st.number_input("Indice UVB", value=5.0)
                tab_data['CLOUD_AMT'] = st.number_input("Couverture Nuageuse", value=10.0)
            with col_b:
                tab_data['SOIL_RW'] = st.number_input("Humidité Sol (SOIL_RW)", value=0.3)
                tab_data['PRECT_TOTAL'] = st.number_input("Précipitations (PRECT_TOTAL)", value=12.5)
                tab_data['SURFACE_P'] = st.number_input("Pression Surface", value=1012.3)
                tab_data['SH2M'] = st.number_input("Humidité (SH2M)", value=15.0)

        if st.button("Lancer le Diagnostic Hybride", type="primary"):
            with st.spinner("Exécution des modèles en cours..."):
                probs_finales = None
                
                # Cas 1 : Image seule
                if image is not None and not use_tabular:
                    probs_finales = predire_avec_cnn(image)
                    st.success("Analyse effectuée via CNN (U-Net) uniquement.")
                
                # Cas 2 : Tableau seul
                elif image is None and use_tabular:
                    probs_finales = predire_avec_xgboost(tab_data)
                    st.success("Analyse effectuée via XGBoost (Tabulaire) uniquement.")
                
                # Cas 3 : Les deux (Fusion / Soft Voting)
                elif image is not None and use_tabular:
                    p_cnn = predire_avec_cnn(image)
                    p_xgb = predire_avec_xgboost(tab_data)
                    probs_finales = (p_cnn + p_xgb) / 2.0
                    st.success("Analyse hybride réussie (Moyenne CNN + XGBoost) !")
                else:
                    st.error("Veuillez fournir au moins une image ou des données tabulaires.")

                if probs_finales is not None:
                    classe_pred = np.argmax(probs_finales)
                    confiance = probs_finales[classe_prediter if 'classe_prediter' in locals() else classe_pred]
                    
                    if classe_pred == 1 or confiance > 0.5: # Selon ton encoding
                        st.metric(label="Résultat du Modèle", value="Risque / Positif", delta=f"{confiance*100:.1f}% Confiance", delta_color="inverse")
                    else:
                        st.metric(label="Résultat du Modèle", value="Négatif / Normal", delta=f"{(1-confiance)*100:.1f}% Confiance")

                    st.markdown("---")
                    st.markdown("### 🩺 Recommandations Médicales")
                    st.warning("""
                    1. **Consulter un médecin** pour un frottis cutané de confirmation.
                    2. **Protéger la lésion** avec un pansement propre pour éviter la surinfection.
                    """)

# ==========================================
# VOLET 3: Model Explainability (SHAP)
# ==========================================
elif volet == "Volet 3: Model Explainability (SHAP)":
    st.title("🧩 Interprétabilité du Modèle Tabulaire (SHAP)")
    st.write("""
    Cette section explique comment le modèle **XGBoost** prend ses décisions en analysant l'impact de chaque variable environnementale (vent, humidité, précipitations, etc.).
    """)
    
    if xgb_pipeline is not None and df_shap is not None:
        if st.button("Générer l'explication SHAP globale"):
            with st.spinner("Calcul des valeurs de Shapley en cours..."):
                # Préparation des données pour SHAP à travers le pipeline
                scaler = xgb_pipeline.named_steps['scaler']
                pca = xgb_pipeline.named_steps['pca']
                model_xgb = xgb_pipeline.named_steps['xgb']
                
                # Nettoyage des colonnes comme dans l'entraînement
                X_eval = df_shap.drop(columns=['Cases', 'Cases_Class', 'Mois', 'Date'], errors='ignore')
                
                # Transformation via le pipeline partiel
                X_scaled = scaler.transform(X_eval)
                X_pca = pca.transform(X_scaled)
                
                # Calcul des valeurs SHAP
                explainer = shap.TreeExplainer(model_xgb)
                shap_values = explainer(X_pca)
                
                # Affichage du graphique SHAP sous Streamlit
                st.subheader("Impact des Composantes Principales (PCA) sur la prédiction")
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values, X_pca, show=False)
                st.pyplot(fig)
                
                st.info("""
                **Comment lire ce graphique :**
                * Chaque point représente une instance de données.
                * La couleur indique la valeur de la caractéristique (Rouge = Élevée, Bleue = Faible).
                * L'axe horizontal montre si l'effet de cette caractéristique pousse la prédiction vers un risque élevé (droite) ou faible (gauche).
                """)
    else:
        st.warning("Impossible de charger les données ou le pipeline XGBoost pour calculer SHAP. Assurez-vous que `leishmaniasis_pipeline.pkl` et le fichier CSV existent.")