import torch
import src.config as config
import joblib
from tqdm import tqdm
import numpy as np
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt

from models.bert import UIT_VFSC_Dataset as BERTDataset
from models.bert import collate_fn as BERTDataset_collate_fn
from models.lstm import UIT_VFSC_Dataset as LSTMDataset
from models.lstm import collate_fn as LSTMDataset_collate_fn
from models.fasttext_svm import converter

from models.bert import PhoBertFeedForward_base, PhoBERTLSTM_base, PhoBertFeedForward_large, PhoBERTLSTM_large, PhoBERTModel
from models.lstm import FastTextLSTMModel
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

from sklearn import metrics

model_dict = {
    1: "PhoBERT(base) + FeedForward",
    2: "PhoBERT(large) + FeedForward" ,
    3: "PhoBERT(base) + LSTM",
    4: "PhoBERT(large) + LSTM",
    5: "FastText + LSTM",
    6: "FastText + SVM"
}

def evaluate(y_true, y_pred, labels=None):
    print(f"Precision: {metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=1)}")
    print(f"Recall: {metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted')}")
    print(f"F1 score: {metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')}")

    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

def ensemble_fn(pred1, pred2, ratio):
  return ratio * pred1 + (1-ratio) * pred2

def load_PhoBERT(model_idx):
    phobert_testdata = BERTDataset(config.TEST_PATH)
    phobert_testdataloader = DataLoader(dataset=phobert_testdata, batch_size=config.BATCH_SIZE,
                                        collate_fn=BERTDataset_collate_fn, shuffle=False,num_workers=config.NUM_WORKERS)

    if model_idx == 1:
        model = PhoBertFeedForward_base(from_pretrained=False)
        ckpt_path = config.PHOBERT_base_FF_CKPT
    if model_idx == 2:
        model = PhoBertFeedForward_large(from_pretrained=False)
        ckpt_path = config.PHOBERT_large_FF_CKPT
    if model_idx == 3:
        model = PhoBERTLSTM_base(from_pretrained=False)
        ckpt_path = config.PHOBERT_base_LSTM_CKPT
    if model_idx == 4:
        model = PhoBERTLSTM_large(from_pretrained=False)
        ckpt_path = config.PHOBERT_large_LSTM_CKPT

    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    system = PhoBERTModel(model)
    system.load_state_dict(ckpt["state_dict"])
    system.eval()
    system.to(config.DEVICE)

    prediction = []

    with torch.no_grad():
        loop = tqdm(phobert_testdataloader)
        for input_ids, attn_mask, label in loop:
            pred = system(input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE))
            prediction.append(pred.cpu().numpy())
            loop.set_description("Working")

    prediction_final = np.concatenate(prediction, axis=0)
    return prediction_final

def load_FastText_LSTM():
    fast_lstm_testdata = LSTMDataset(config.TEST_PATH)
    fast_lstm_testdataloader = DataLoader(dataset=fast_lstm_testdata, batch_size=config.BATCH_SIZE,
                                        collate_fn=LSTMDataset_collate_fn, shuffle=False, num_workers=config.NUM_WORKERS)
    fast_lstm_system = FastTextLSTMModel()
    fast_ckpt = torch.load(config.FAST_LSTM_CKPT, map_location=config.DEVICE)
    fast_lstm_system.load_state_dict(fast_ckpt['state_dict'])
    fast_lstm_system.eval()
    fast_lstm_system.to(config.DEVICE)

    fast_lstm_prediction = []

    with torch.no_grad():
        loop=tqdm(fast_lstm_testdataloader)
        for vec, label in loop:
            pred=fast_lstm_system(vec.to(config.DEVICE))
            fast_lstm_prediction.append(pred.cpu().numpy())

            loop.set_description("Working")

    fast_lstm_final = np.concatenate(fast_lstm_prediction, axis=0)
    return fast_lstm_final

def load_SVM():
    test_df = pd.read_csv(config.TEST_PATH, sep="\t")
    X_test = np.array(test_df["sents"])
    X_test_vec = np.array([converter.text_to_vector(text) for text in X_test])

    svm = joblib.load(config.FAST_SVM_CKPT)
    svm_pred = svm.predict_proba(X_test_vec)
    return svm_pred

def LOAD_MODEL(model_idx):
    if model_idx in range(1,5):
        return load_PhoBERT(model_idx)
    if model_idx == 5:
        return load_FastText_LSTM()
    else: 
        return load_SVM()

if __name__ == "__main__":
    print(colored("\tEnsemble evaluation on 2 models", "blue"))
    print("List of models:\n")
    print("(1)PhoBERT(base) + FeedForward")
    print("(2)PhoBERT(large) + FeedForward")
    print("(3)PhoBERT(base) + LSTM")
    print("(4)PhoBERT(large) + LSTM")
    print("(5)FastText + LSTM")
    print("(6)FastText + SVM")


    model1_idx = int(input("Choose 1st model: "))
    model2_idx = int(input("Choose 2nd model: "))

    print(f"Ensemble evaluation on {model_dict[model1_idx]} & {model_dict[model2_idx]}")
    pred1 = LOAD_MODEL(model1_idx)
    pred2 = LOAD_MODEL(model2_idx)

    test_df = pd.read_csv(config.TEST_PATH, sep="\t")
    y_test = test_df[config.LABEL]

    ensemble_result = ensemble_fn(pred1, pred2, config.RATIO)
    pred = np.argmax(ensemble_result, axis=1)
    evaluate(y_test, pred)