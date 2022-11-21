import torch

NUM_CLASSES = 3
LABEL = "sentiments"

CHECKPOINT = "vinai/phobert-base"
FAST_TEXT_PRETRAINED = "FastText_embedding/fasttext_train_dev.model"
MAX_LENGTH = 16

# DATA
TRAIN_PATH = "./uit_vsfc_data/train.csv"
VALIDATION_PATH = "./uit_vsfc_data/dev.csv"
TEST_PATH = "./uit_vsfc_data/test.csv"

PHOBERT_base_FF_CKPT = None
PHOBERT_base_LSTM_CKPT = None
PHOBERT_large_FF_CKPT = None
PHOBERT_large_LSTM_CKPT = None
FAST_LSTM_CKPT = None
FAST_SVM_CKPT = None

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 2

# ENSEMBLE
RATIO = 0.5