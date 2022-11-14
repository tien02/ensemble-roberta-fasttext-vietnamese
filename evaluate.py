import joblib
import torch
import config
from tqdm import tqdm
import numpy as np
from termcolor import colored

from models.phobert import UIT_VFSC_Dataset as BERTDataset
from models.phobert import collate_fn as BERTDataset_collate_fn
from models.fasttext_lstm import UIT_VFSC_Dataset as LSTMDataset
from models.fasttext_lstm import collate_fn as LSTMDataset_collate_fn
from models.fasttext_svm import converter

from models.phobert import PhoBertFeedForward_base, PhoBERTLSTM_base, PhoBERTModel
from models.fasttext_lstm import FastTextLSTMModel
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

def ensemble_fn(pred1, pred2):
    return pred1 * 0.5 + pred2 * 0.5

if __name__ == "__main__":
    accuracy_fn = Accuracy(num_classes=config.NUM_CLASSES, average="weighted").to(config.DEVICE)
    precision_fn = Precision(num_classes=config.NUM_CLASSES, average="weighted").to(config.DEVICE)
    recall_fn = Recall(num_classes=config.NUM_CLASSES, average="weighted").to(config.DEVICE)
    f1_fn = F1Score(num_classes=config.NUM_CLASSES, average="weighted").to(config.DEVICE)

    phobert_testdata = BERTDataset(config.TEST_PATH)
    phobert_testdataloader = DataLoader(dataset=phobert_testdata, batch_size=config.BATCH_SIZE,
                                        collate_fn=BERTDataset_collate_fn, shuffle=False,num_workers=config.NUM_WORKERS)

    fasttext_lstm_testdata = LSTMDataset(config.TEST_PATH)
    fasttext_lstm_testdataloader = DataLoader(dataset=fasttext_lstm_testdata, batch_size=config.BATCH_SIZE,
                                        collate_fn=LSTMDataset_collate_fn, shuffle=False,num_workers=config.NUM_WORKERS)

    phobert_ff_ckpt = torch.load(config.PHOBERT_FF_CKPT, map_location=config.DEVICE)
    fasttext_lstm_ckpt = torch.load(config.LSTM_CKPT, map_location=config.DEVICE)
    phobert_lstm_ckpt = torch.load(config.PHOBERT_LSTM_CKPT, map_location=config.DEVICE)

    phobert_ff = PhoBertFeedForward_base(from_pretrained=False)
    phobert_lstm = PhoBERTLSTM_base(from_pretrained=False)

    phobert_ff_system = PhoBERTModel(phobert_ff)
    phobert_ff_system.load_state_dict(phobert_ff_ckpt["state_dict"])
    phobert_ff_system.eval()
    phobert_ff_system.to(config.DEVICE)

    phobert_lstm_system = PhoBERTModel(phobert_lstm)
    phobert_lstm_system.load_state_dict(phobert_lstm_ckpt["state_dict"])
    phobert_lstm_system.eval()
    phobert_lstm_system.to(config.DEVICE)

    # fasttext_lstm_system = FastTextLSTMModel()
    # fasttext_lstm_system.load_state_dict(fasttext_lstm_ckpt["state_dict"])
    # fasttext_lstm_system.eval()
    # fasttext_lstm_system.to(config.DEVICE)

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    with torch.no_grad():
        loop = tqdm(phobert_testdataloader)
        for input_ids, attn_mask, label in loop:
            phobert_ff_pred = phobert_ff_system(input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE))
            phobert_lstm_pred = phobert_lstm_system(input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE))
            
            ensemble_pred = ensemble_fn(phobert_ff_pred, phobert_lstm_pred)

            accuracy_val = accuracy_fn(ensemble_pred, label.to(config.DEVICE))
            precision_val = precision_fn(ensemble_pred, label.to(config.DEVICE))
            recall_val = recall_fn(ensemble_pred, label.to(config.DEVICE))
            f1_val = f1_fn(ensemble_pred, label.to(config.DEVICE))

            acc_list.append(accuracy_val)
            precision_list.append(precision_val)
            recall_list.append(recall_val)
            f1_list.append(f1_val)

            loop.set_description("Working")

    # with torch.no_grad():
    #     loop1 = tqdm(phobert_testdataloader)
    #     loop2 = tqdm(fasttext_lstm_testdataloader)

    #     pred1 = []
    #     pred2 = []
    #     label1 = []
    #     label2 = []

    #     for input_ids, attn_mask, label in loop1:
    #         phobert_ff_pred = phobert_ff_system(input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE))
    #         pred1.append(phobert_ff_pred)
    #         label1.append(label)
    #         loop1.set_description("Working on Model 1")


    #     for vec, label in loop2:
    #         fasttext_lstm_pred = fasttext_lstm_system(vec.to(config.DEVICE))
    #         pred2.append(fasttext_lstm_pred)
    #         label2.append(label)
    #         loop2.set_description("Working on Model 2")

    #     # assert label1 == label2, "2 dataloader not in order."

    #     ensemble_pred = ensemble_fn(torch.tensor(pred1), torch.tensor(pred2))
    #     label1 = torch.tensor(label1)

    #     accuracy_val = accuracy_fn(ensemble_pred, label1)
    #     precision_val = precision_fn(ensemble_pred, label1)
    #     recall_val = recall_fn(ensemble_pred, label1)
    #     f1_val = f1_fn(ensemble_pred, label1)

    #     print(colored(f"Accuracy: {accuracy_val:.4f}", "blue"))        
    #     print(colored(f"Precision: {precision_val:.4f}", "blue"))        
    #     print(colored(f"Recall: {recall_val:.4f}", "blue"))        
    #     print(colored(f"F1-score: {f1_val:.4f}", "blue"))        
        

    print(colored(f"Accuracy: {np.mean(acc_list):.4f}", "blue"))        
    print(colored(f"Precision: {np.mean(precision_list):.4f}", "blue"))        
    print(colored(f"Recall: {np.mean(recall_list):.4f}", "blue"))        
    print(colored(f"F1-score: {np.mean(f1_list):.4f}", "blue"))        