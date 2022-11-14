import torch

NUM_CLASSES = 3

CHECKPOINT = "vinai/phobert-base"
FAST_TEXT_PRETRAINED = "FastText_embedding/fasttext_train_dev.model"
MAX_LENGTH = 16

# DATA
TRAIN_PATH = "./uit_vsfc_data/train.csv"
VALIDATION_PATH = "./uit_vsfc_data/dev.csv"
TEST_PATH = "./uit_vsfc_data/test.csv"

PHOBERT_FF_CKPT = "UIT_VFSC_checkpoint/phobert_ckpt/feedforward/epoch=15-step=2864.ckpt"
PHOBERT_LSTM_CKPT = "UIT_VFSC_checkpoint/phobert_ckpt/lstm/epoch=19-step=3580.ckpt"
LSTM_CKPT = "UIT_VFSC_checkpoint/fasttext_ckpt/lstm/epoch=115-step=20764.ckpt"
SVM_CKPT = "UIT_VFSC_checkpoint/fasttext_ckpt/svm/fast_svm_sentiments.pkl"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = torch.device("cpu")
BATCH_SIZE = 8
NUM_WORKERS = 2