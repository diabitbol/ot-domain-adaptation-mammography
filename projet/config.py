# ==============================================================================
# CONFIGURATION : Génération des données et Entraînement
# ==============================================================================

# --- Sauvegarde et Chemins ---
NOM_MODELE_SAUVEGARDE = "new2_best_model_700"
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
# 0: Pas de normalisation | 1: Par image | 2: Sur le dataset global
CHOIX_NORMALISATION_IMAGE = 0
# Seuil pour exclure le fond noir de la normalisation
LIMITE_BASSE = 0.05

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
