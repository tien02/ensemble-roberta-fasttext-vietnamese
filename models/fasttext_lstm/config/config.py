import os

# DATA PATH
TRAIN_PATH = os.path.abspath("uit_vsfc_data/train.csv")
VALIDATION_PATH = os.path.abspath("uit_vsfc_data/dev.csv")
TEST_PATH = os.path.abspath("uit_vsfc_data/test.csv")

# DATASET CONFIG
LABEL = "sentiments"
FAST_TEXT_PRETRAINED = os.path.abspath("FastText_embedding/fasttext_train_dev.model")

# DATALOADERCONFIG
BATCH_SIZE = 2
NUM_WORKERS = 8

# MODEL
LEARNING_RATE = 1e-4
VECTOR_SIZE = 300
DROP_OUT = 0.4
OUT_CHANNELS = 3

# TENSORBOARD LOG
TENSORBOARD = {
    "DIR": "LOG",
    "NAME": f"LSTM_FASTTEXT_VEC{VECTOR_SIZE}_DROP{DROP_OUT}_LABEL{LABEL}_LR{LEARNING_RATE}",
    "VERSION": "0",
}

# CHECKPOINT
CHECKPOINT_DIR = os.path.join(TENSORBOARD["DIR"], TENSORBOARD["NAME"], TENSORBOARD["VERSION"], "CKPT")

# TRAINER
ACCELERATOR = "gpu"
NUM_EPOCHS = 300
EVAL_EVERY_EPOCHS = 4

# EVALUATE
TEST_CKPT_PATH = "checkpoint/epoch=99-step=17900.ckpt"

# KEEP_TRAINING
CONTINUE_TRAINING = None