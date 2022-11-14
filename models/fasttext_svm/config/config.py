import os

# DATA PATH
TRAIN_PATH = "../../uit_vsfc_data/train.csv"
VALIDATION_PATH = "../../uit_vsfc_data/dev.csv"
TEST_PATH = "../../uit_vsfc_data/test.csv"
LABEL = "sentiments"

# EMBEDDING CONFIG
MAX_LENGTH = 16
EMBEDDING_PATH = os.path.abspath("FastText_embedding/fasttext_train_dev.model")

# CACHE
CACHE_NAME = "fast_svm_cache"

# SAVE_FILE
SAVE_PATH = f"fast_svm_{LABEL}.pkl"

# MODEL
PCA_COMPONENTS = 800