import os
import torch
import joblib
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# 1. CONFIGURATION ET CHARGEMENT DES MODÈLES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = "timm-mobilenetv3_small_100"
PATCH_SIZE = 128

print("Chargement des modèles...")

# A. Charger le Pipeline XGBoost (Tabulaire)
chemin_xgb_pipeline = 'leishmaniasis_pipeline.pkl'
if os.path.exists(chemin_xgb_pipeline):
    xgb_pipeline = joblib.load(chemin_xgb_pipeline)
    print("[OK] Pipeline XGBoost chargé.")
else:
    xgb_pipeline = None
    print("[Attention] Fichier XGBoost introuvable. Seul le CNN fonctionnera.")

# B. Charger le CNN U-Net (Image)
cnn_model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

chemin_unet_weights = 'weights/unet_best.pth'
if os.path.exists(chemin_unet_weights):
    cnn_model.load_state_dict(torch.load(chemin_unet_weights, map_location=DEVICE))
    cnn_model.eval()
    print("[OK] Poids du U-Net chargés.")
else:
    print("[Erreur] Fichier de poids 'weights/unet_best.pth' introuvable !")

# Transformations d'images (identiques à la validation)
transformations_inference = A.Compose([
    A.Resize(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ==========================================
# 2. FONCTIONS DE PRÉDICTION SÉPARÉES
# ==========================================

def predire_avec_cnn(image_path):
    """Traite la photo via le CNN et retourne les probabilités [Negatif, Positif]."""
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Impossible de lire l'image à l'emplacement : {image_path}")
        
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Préparation du tenseur
    tenseur_image = transformations_inference(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = cnn_model(tenseur_image)
        probs_pixels = torch.sigmoid(logits)
        
        # Réduction de la carte de segmentation 2D en une probabilité globale unique (Max Pooling)
        prob_lesion = probs_pixels.max().item()
        
    # Formatage sous forme de tableau [Prob_Classe_0, Prob_Classe_1]
    return np.array([1.0 - prob_lesion, prob_lesion])


def predire_avec_xgboost(donnees_tabulaires):
    """Traite les données tabulaires via XGBoost et retourne les probabilités."""
    if xgb_pipeline is None:
        raise RuntimeError("Le pipeline XGBoost n'a pas été chargé.")
        
    if isinstance(donnees_tabulaires, dict):
        df = pd.DataFrame([donnees_tabulaires])
    else:
        df = donnees_tabulaires
        
    # predict_proba renvoie les probabilités pour les deux classes
    return xgb_pipeline.predict_proba(df)[0]


# ==========================================
# 3. ORCHESTRATEUR PRINCIPAL (LA COMBINAISON)
# ==========================================

def predire_global(image_path, donnees_tabulaires=None):
    """
    - Cas 1 : Photo seule -> Utilise uniquement le CNN.
    - Cas 2 : Photo + Données tabulaires -> Traite séparément puis fait la moyenne.
    """
    print(f"\n--- Lancement de l'analyse pour : {image_path} ---")
    
    # 1. Traitement de la photo par le CNN
    probs_cnn = predire_avec_cnn(image_path)
    
    # 2. Vérification des données tabulaires
    if donnees_tabulaires is None or len(donnees_tabulaires) == 0:
        print("Mode : Photo seule (CNN uniquement).")
        probs_finales = probs_cnn
    else:
        print("Mode : Photo + Données tabulaires (Combinaison CNN + XGBoost).")
        probs_xgb = predire_avec_xgboost(donnees_tabulaires)
        
        # Calcul de la moyenne des prédictions (Soft Voting)
        print("Calcul de la moyenne des prédictions...")
        probs_finales = (probs_cnn + probs_xgb) / 2.0
        
    # 3. Décision finale (Classe avec la probabilité maximale)
    classe_predite = np.argmax(probs_finales)
    confiance = probs_finales[classe_predite]
    
    return classe_predite, confiance, probs_finales


# ==========================================
# EXEMPLES D'UTILISATION (TESTS)
# ==========================================
if __name__ == "__main__":
    # Remplace par un vrai chemin d'image de test dans ton projet
    exemple_image = "data/raw/segmentation/images/pos_web_000.jpg"
    
    if os.path.exists(exemple_image):
        # TEST 1 : Seulement la photo
        classe, conf, probs = predire_global(image_path=exemple_image)
        print(f">> Résultat (CNN seul) -> Classe : {classe} | Confiance : {conf * 100:.2f}% | Probs : {probs}")
        
        # TEST 2 : Photo + Données Tabulaires (Dictionnaire avec les colonnes de ton modèle)
        exemple_tableau = {
            'Annee': 2026, 
            'WS10M': 2.5, 
            'WD2M': 120.0, 
            'UVB': 5.0, 
            'CLOUD_AMT': 10.0, 
            'SOIL_RW': 0.3, 
            'PRECT_TOTAL': 12.5, 
            'SURFACE_P': 1012.3, 
            'SH2M': 15.0
        }
        
        classe, conf, probs = predire_global(image_path=exemple_image, donnees_tabulaires=exemple_tableau)
        print(f">> Résultat (Combiné)    -> Classe : {classe} | Confiance : {conf * 100:.2f}% | Probs : {probs}")
    else:
        print(f"Image de test introuvable à l'emplacement : {exemple_image}. Modifie le chemin pour tester.")