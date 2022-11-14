import torch
import torch.nn as nn
import pandas as pd
from .config import config
from torch.utils.data import Dataset, DataLoader
from transformers import PhobertTokenizer
from utils import preprocess_fn
from termcolor import colored

tokenizer = PhobertTokenizer.from_pretrained(config.CHECKPOINT)

class UIT_VFSC_Dataset(Dataset):
    def __init__(self, root_dir, label="sentiments"):
        self.dataframe = pd.read_csv(root_dir, sep="\t")
        self.label = label
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        df = self.dataframe.iloc[index]
        X = df["sents"]
        y = df[self.label]

        x_tokens = preprocess_fn(X)
        tokens = tokenizer(x_tokens)

        return torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"]), torch.tensor(y)

def collate_fn(batch):
    input_ids_list, attn_mask_list, label_list = [], [], []
    for input_ids, attn_mask, label in batch:
        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
        label_list.append(label)
    label_list = torch.tensor(label_list)
    
    input_ids_list = nn.utils.rnn.pad_sequence(input_ids_list)
    attn_mask_list = nn.utils.rnn.pad_sequence(attn_mask_list)

    input_ids_list = torch.permute(input_ids_list, (1, 0))
    attn_mask_list = torch.permute(attn_mask_list, (1, 0))
    
    return input_ids_list, attn_mask_list, label_list

def test():
    data = UIT_VFSC_Dataset(root_dir=config.TRAIN_PATH, label=config.LABEL)
    dataloader = DataLoader(dataset=data, shuffle=False, collate_fn=collate_fn, batch_size=config.BATCH_SIZE)
    for input_ids, attn_mask, _ in dataloader:
        input_ids_len = input_ids.size(0)
        attn_mask_len = attn_mask.size(0)
        break
    assert input_ids_len == attn_mask_len, "input_ids and attention_mask should equal"
    print(colored("Pass", "green"))

if __name__ == "__main__":
    test()