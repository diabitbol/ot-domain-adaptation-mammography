# ==============================================================================
# CONFIGURATION : importation des donnees depuis le S3
# ==============================================================================

import os


# --- Chemin dossier ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Chemins de données ---
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "data")
# Le dossier source DDSM sur le S3
BUCKET_S3_DDSM = "s3/lucasvital/stat_app/nouveau_dataset_mini_ddsm_700_700/"
# Le dossier de destination sur le disque local de l'instance Onyxia
DDSM_PATH_LOCAL = os.path.join(LOCAL_DATA_PATH, "nouveau_dataset_mini_ddsm_700_700")

# Le dossier source VinDr sur le S3
BUCKET_S3_VINDR_NORMAL = "s3/lucasvital/stat_app/0-normal/"
BUCKET_S3_VINDR_CANCER_BENIGN = "s3/lucasvital/stat_app/1-cancer_benign/"
# Le dossier de destination sur le disque local de l'instance Onyxia
CHEMIN_DATA_VINDR = os.path.join(LOCAL_DATA_PATH, "dataset_vindr")
VINDR_PATH_LOCAL_NORMAL = os.path.join(CHEMIN_DATA_VINDR, "0-normal")
VINDR_PATH_LOCAL_CANCER_BENIGN = os.path.join(CHEMIN_DATA_VINDR, "1-cancer_benign")


# ==============================================================================
# CONFIGURATION : Génération des données et Entraînement
# ==============================================================================

# --- Sauvegarde et Chemins ---
NOM_MODELE_SAUVEGARDE = "model1_700"
NOM_MODELE_SAUVEGARDE_CLAHE = NOM_MODELE_SAUVEGARDE + "clahe"
NOM_MODELE_SAUVEGARDE_EWC = NOM_MODELE_SAUVEGARDE + "EWC"
RACINE_MODELE_LOCAL = os.path.join(BASE_DIR, "model")
RACINE_MODELE_S3 = "s3/lucasvital/stat_app/"
CHEMIN_MODELE_LOCAL = os.path.join(RACINE_MODELE_LOCAL, NOM_MODELE_SAUVEGARDE + ".pth")
CHEMIN_MODELE_LOCAL_CLAHE = os.path.join(RACINE_MODELE_LOCAL, NOM_MODELE_SAUVEGARDE_CLAHE + ".pth")
CHEMIN_MODELE_LOCAL_EWC = os.path.join(RACINE_MODELE_LOCAL, NOM_MODELE_SAUVEGARDE_EWC + ".pth")
CHEMIN_MODELE_S3 = os.path.join(RACINE_MODELE_S3, NOM_MODELE_SAUVEGARDE + ".pth")
CHEMIN_MODELE_S3_CLAHE = os.path.join(RACINE_MODELE_S3, NOM_MODELE_SAUVEGARDE_CLAHE + ".pth")
CHEMIN_MODELE_S3_EWC = os.path.join(RACINE_MODELE_S3, NOM_MODELE_SAUVEGARDE_EWC + ".pth")

# NOTE : Penser à vérifier le chemin S3 dans le script d'entraînement

# --- Paramètres des images ---
RESOLUTION = 700
BATCH_SIZE = 32

# --- Data Augmentation (Bruit et Variations) ---
# Transformation de +/- 10% selon une loi uniforme
CONTRAST = 0.1
BRIGHTNESS = 0.1
# Rotation de +/- 15 degrés (bords noirs créés)
DEGREES = 15

# --- Normalisation ---
# 0: Pas de normalisation | 1: Par image | 2: Sur le dataset global # à détailler...
CHOIX_FONCTION = 0
# Seuil pour exclure le fond noir de la normalisation
LIMITE_BASSE = 0.05
# Choix de normaliser les images :
#  ie transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
# pour TEST_SIZE = 1
NORMALIZE = 1

# --- Division du Dataset (Splitting) ---
NUMERO_RANDOM = 42
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
# METHODE_GENERATION = "surepresentation"  # Optionnel : surep / sous-rep

# --- Hyperparamètres d'entraînement ---
DROPOUT = 0.5  # Taux de désactivation neurale
LR = 0.0001  # Learning Rate initial
WEIGHT_DECAY = 1e-4  # Régularisation L2 pour éviter l'overfitting

# --- Scheduler (Gestion du LR pendant l'entraînement) ---
FACTOR = 0.1  # Facteur de réduction du LR
PATIENCE = 5  # Attente du scheduler avant réduction
THRESHOLD = 0.0001  # Seuil de progression minimale

# --- Early Stopping ---
PATIENCE_LIMITE = 12  # Arrêt si pas d'amélioration de la Val Loss après 12 époques
NUM_EPOCHS = 80  # Nombre maximum d'époques

# --- EWC (Elastic Weight Consolidation) ---
LAMBDA_EWC = 1000  # Importance accordée à la tâche précédente (DDSM)
