from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASETS_ROOT = Path(r"/Workspace/Projects/Datasets")

GLOVE_PATH = DATASETS_ROOT / "Embeddings/GloVe/glove.6B.100d.txt"

DATASET_NAME = "SMSSpamCollection"

TEXT_COLUMN = "message"
LABEL_COLUMN = "label"

SEED = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

SEQUENCE_LENGTH = 50
MAX_TOKENS = 10000