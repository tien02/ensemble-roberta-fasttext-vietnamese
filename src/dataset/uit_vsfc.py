import os

import torch
from torch.utils.data import Dataset

from transformers import PhobertTokenizer

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.utils.class_weight import compute_class_weight

from .utils import preprocess_fn


class UIT_VSFC(Dataset):
    def __init__(self, data_dir:str, label:str = "sentiments", model_type:str = "bert", fasttext_embedding:str = None):
        '''
        Parameters:
            data_dir (str): path to data directory
            label (str) : Type of label, support "sentiments" or "topics"
            model_type (str): Type of model, support "bert" or "lstm"
        '''
        self.features = pd.read_table(os.path.join(data_dir, "sents.txt"), names=['sents'])

        assert label in ["sentiments", "topics"], f"Expect 'label' argument to be 'sentiments' or 'topics', unknown '{label}'."
        if label == "sentiments":
            self.labels = pd.read_table(os.path.join(data_dir, "sentiments.txt"), names=["labels"])
        if label == "topics":
            self.labels = pd.read_table(os.path.join(data_dir, "topics.txt"), names=["labels"])
        
        assert model_type in ['bert', 'lstm'], f"Expect 'model_type' argument to be 'bert' or 'lstm', unknown '{model_type}'."
        self.model_type = model_type

        if self.model_type == 'lstm':
            self.word_vec = KeyedVectors.load(fasttext_embedding)

        self.tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")

        y = self.labels.values
        self.class_weights = torch.tensor(compute_class_weight(class_weight="balanced",classes=np.unique(np.ravel(y)),y=np.ravel(y)),dtype=torch.float)
        

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        X = self.features.iloc[index].values[0]
        y = self.labels.iloc[index].values[0]
        x_tokens = preprocess_fn(X)
        
        if self.model_type == "bert":
            tokens = self.tokenizer(x_tokens)
            return torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"]), torch.tensor(y)
        else:
            x_embed = []
            for x_token in x_tokens:
                try:
                    x_embed.append(torch.unsqueeze(torch.tensor(self.word_vec.wv[x_token]), dim=0))
                except Exception as e:
                    pass
            return torch.cat(x_embed, dim=0), torch.tensor(y)