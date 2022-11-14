import os

# PRETRAINED MODEL
CHECKPOINT = "vinai/phobert-base"

# DATA
TRAIN_PATH = "../../uit_vsfc_data/train.csv"
VALIDATION_PATH = "../../uit_vsfc_data/dev.csv"
TEST_PATH = "../../uit_vsfc_data/test.csv"

# DATALOADER
LABEL = "sentiments"
NUM_WORKERS = 3

# TRAINER
EPOCHS = 3
VAL_EACH_EPOCH = 1
NUM_CLASSES = 3
BATCH_SIZE = 64
THRESHOLD=0.5
LEARNING_RATE = 1e-4
ACCELERATOR = "cpu"

# MODEL
SEED = 42
MODEL = "FeedForward-base"   # "FeedForward"/"LSTM" + '-' +  'base'/'large'
DROP_OUT = 0.4

# TENSORBOARD LOG
TENSORBOARD = {
    "DIR": "LOG",
    "NAME": f"{MODEL}_DROP{DROP_OUT}_LABEL{LABEL}_LR{LEARNING_RATE}",
    "VERSION": "0",
}

# CHECKPOINT
CHECKPOINT_DIR = os.path.join(TENSORBOARD["DIR"], TENSORBOARD["NAME"], TENSORBOARD["VERSION"], "CKPT")

# EVALUATE
TEST_CKPT_PATH = "checkpoint/epoch=23-step=4296.ckpt"

# KEEP_TRAINING PATH
CONTINUE_TRAINING = None