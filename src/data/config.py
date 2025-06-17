# /config.py
import torch

# === DATA CONFIGURATION ===
# Path di default per il dataset Food101. Verrà creato se non esiste.
# Può essere sovrascritto da riga di comando con l'argomento --data_dir.
DEFAULT_DATA_DIR = "food101_split"

# Parametri specifici per la preparazione automatica di Food101
FOOD101_NUM_CLASSES = 100
FOOD101_MAX_PER_CLASS = 250

# === GENERAL CONFIGURATION ===
RANDOM_SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === MODEL CONFIGURATION ===
MODEL_NAME = "ViT-L-14"
PRETRAINED_WEIGHTS = "openai"
EMBED_DIM = 2048
UNFREEZE_LAYERS = 4

# === TRAINING CONFIGURATION ===
EPOCHS = 12
WARMUP_EPOCHS = 5
BATCH_SIZE = 32
LR_BASE = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4

# === LOSS CONFIGURATION ===
PROXY_MARGIN = 0.2
PROXY_ALPHA = 64
TRIPLET_MARGIN = 0.2
CE_WEIGHT = 1.0

# === EVALUATION CONFIGURATION ===
TOP_K_VALUES = [1, 5, 10]
