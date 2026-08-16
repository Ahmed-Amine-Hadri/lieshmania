import streamlit as st
from PIL import Image
import os
import sys
import os

# 1. Ajouter le dossier racine du projet au chemin système de Python
chemin_racine = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if chemin_racine not in sys.path:
    sys.path.append(chemin_racine)

import streamlit as st
from PIL import Image

# 2. Maintenant Python sait où trouver le dossier 'src' !
from src.predict import predire_global
from src.interpretability import afficher_shap

# --- Le reste de ton code reste inchangé ci-dessous ---
st.set_page_config(page_title="Leishmaniose - IA Expert", layout="wide", initial_sidebar_state="expanded")
# ...
# Imports depuis le dossier src
from src.predict import predire_global
from src.interpretability import afficher_shap

st.set_page_config(page_title="Leishmaniose - IA Expert", layout="wide", initial_sidebar_state="expanded")

if 'demarre' not in st.session_state:
    st.session_state['demarre'] = False

if not st.session_state['demarre']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Système Expert IA : Détection et Prédiction de la Leishmaniose</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #5D6D7E;'>Réalisé par Ahmed Amine Hadri</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Commencer", use_container_width=True):
            st.session_state['demarre'] = True
            st.rerun()
else:
    st.sidebar.title("Navigation")
    volet = st.sidebar.radio("Aller à :", ["Présentation du problème", "Prédiction et Interprétabilité"])

    if volet == "Présentation du problème":
        st.title("Volet 1 : La Leishmaniose Cutanée")
        st.header("1. Introduction et Impact sur le Monde")
        st.write("La leishmaniose est une maladie parasitaire causée par le protozoaire *Leishmania*, transmis par la piqûre d'insectes vecteurs appelés phlébotomes. L'OMS estime qu'entre **700 000 et 1 million de nouveaux cas** surviennent chaque année. Les cicatrices engendrent une forte stigmatisation sociale.")
        
        st.header("2. Le Problème Spécifique à Ouarzazate")
        st.write("Dans la région de **Ouarzazate**, l'urbanisation, les barrages et les variations climatiques exacerbent la prolifération des phlébotomes et la transmission zoonotique.")
        
        st.header("3. Statistiques")
        st.write("Les données (2010-2022) montrent une forte corrélation entre le nombre de cas et des variables comme la précipitation (`PRECT_TOTAL`), l'humidité (`RH2M`) et le rayonnement (`UVB`).")

    elif volet == "Prédiction et Interprétabilité":
        st.title("Volet 2 : Prédiction IA & Interprétabilité (SHAP)")
        
        col_img, col_data = st.columns(2)
        
        with col_img:
            st.subheader("1. Image de la Lésion")
            uploaded_file = st.file_uploader("Chargez une image (.jpg, .png)", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Image chargée', use_container_width=True)
                image_path = "temp_image.jpg"
                image.save(image_path)
            else:
                image_path = None

        with col_data:
            st.subheader("2. Données Climatiques")
            annee = st.number_input("Année", value=2024)
            ws10m = st.number_input("Vitesse du vent à 10m", value=3.5)
            wd2m = st.number_input("Direction du vent à 2m", value=180.0)
            uvb = st.number_input("Rayonnement UVB", value=8.2)
            cloud_amt = st.number_input("Couverture Nuageuse", value=25.0)
            soil_rw = st.number_input("Humidité du sol", value=0.4)
            prect_total = st.number_input("Précipitations", value=5.0)
            surface_p = st.number_input("Pression en surface", value=1005.1)
            sh2m = st.number_input("Humidité spécifique à 2m", value=12.0)
            
            donnees_tabulaires = {
                'Annee': annee, 'WS10M': ws10m, 'WD2M': wd2m, 'UVB': uvb, 
                'CLOUD_AMT': cloud_amt, 'SOIL_RW': soil_rw, 'PRECT_TOTAL': prect_total, 
                'SURFACE_P': surface_p, 'SH2M': sh2m
            }

        st.markdown("---")
        
        if st.button("Lancer la Fusion (U-Net + XGBoost) et Analyser", use_container_width=True):
            if image_path is None:
                st.warning("Veuillez charger une image.")
            else:
                with st.spinner("Exécution de l'IA en cours..."):
                    classe, conf, probs = predire_global(image_path, donnees_tabulaires)
                    st.success("Analyse terminée !")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Classe Prédite", "Positif" if classe == 1 else "Négatif")
                    with res_col2:
                        st.metric("Confiance", f"{conf * 100:.2f}%")
                
                st.markdown("---")
                afficher_shap(donnees_tabulaires)