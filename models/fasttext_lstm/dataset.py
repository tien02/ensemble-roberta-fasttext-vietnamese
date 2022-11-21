from .config import config
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from .utils import preprocess_fn
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from sklearn.utils.class_weight import compute_class_weight

word_vec = KeyedVectors.load(config.FAST_TEXT_PRETRAINED)

class UIT_VFSC_Dataset(Dataset):
    def __init__(self, root_dir, label="sentiments"):
        self.dataframe = pd.read_csv(root_dir, sep="\t")
        self.label = label
        y = self.dataframe[self.label].values
        self.class_weights = torch.tensor(compute_class_weight(class_weight="balanced",classes=np.unique(y),y=y),dtype=torch.float)
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        df = self.dataframe.iloc[index]
        X = df["sents"]
        y = df[self.label]

        x_tokens = preprocess_fn(X)
        x_embed = []

        for x_token in x_tokens:
            try:
                x_embed.append(torch.unsqueeze(torch.tensor(word_vec.wv[x_token]), dim=0))
            except Exception as e:
                pass

        return torch.cat(x_embed, dim=0), torch.tensor(y)

def collate_fn(batch):
    vectors_list, label_list = [], []
    for text, label in batch:
        vectors_list.append(text)
        label_list.append(label)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    vectors_list = nn.utils.rnn.pad_sequence(vectors_list)

    return vectors_list, label_list

if __name__ == "__main__":
    data = UIT_VFSC_Dataset(root_dir=config.TRAIN_PATH)
    print(f"Data length: {len(data)}")
    tensor, label = data[0]
    print(f"Tensor shape: {tensor.shape}")