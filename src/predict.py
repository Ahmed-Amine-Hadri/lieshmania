import os
import torch
import joblib
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = "timm-mobilenetv3_small_100"
PATCH_SIZE = 128

# Récupération automatique du chemin racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A. Charger le Pipeline XGBoost
chemin_xgb_pipeline = os.path.join(BASE_DIR, 'leishmaniasis_pipeline.pkl')
if os.path.exists(chemin_xgb_pipeline):
    xgb_pipeline = joblib.load(chemin_xgb_pipeline)
else:
    xgb_pipeline = None

# B. Charger le CNN U-Net
cnn_model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

chemin_unet_weights = os.path.join(BASE_DIR, 'weights', 'unet_best.pth')
if os.path.exists(chemin_unet_weights):
    # LA CORRECTION EST ICI : Chargement robuste du Checkpoint PyTorch
    checkpoint = torch.load(chemin_unet_weights, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        cnn_model.load_state_dict(checkpoint)
    cnn_model.eval()

transformations_inference = A.Compose([
    A.Resize(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predire_avec_cnn(image_path):
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    tenseur_image = transformations_inference(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = cnn_model(tenseur_image)
        probs_pixels = torch.sigmoid(logits)
        prob_lesion = probs_pixels.max().item()
        
    return np.array([1.0 - prob_lesion, prob_lesion])

def predire_avec_xgboost(donnees_tabulaires):
    df = pd.DataFrame([donnees_tabulaires])
    return xgb_pipeline.predict_proba(df)[0]

def predire_global(image_path, donnees_tabulaires=None):
    probs_cnn = predire_avec_cnn(image_path)
    
    if donnees_tabulaires is None or len(donnees_tabulaires) == 0:
        probs_finales = probs_cnn
    else:
        probs_xgb = predire_avec_xgboost(donnees_tabulaires)
        probs_finales = (probs_cnn + probs_xgb) / 2.0
        
    classe_predite = np.argmax(probs_finales)
    confiance = probs_finales[classe_predite]
    
    return classe_predite, confiance, probs_finales