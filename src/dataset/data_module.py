import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .uit_vsfc import UIT_VSFC
from .utils import bert_collate_fn, lstm_collate_fn

class DataModule(pl.LightningDataModule):
    def __init__(self, root_data_dir:str, label:str, model_type:str, batch_size:int, num_workers:int, fasttext_embedding:str=None):
        super().__init__()
        self.root_data_dir = root_data_dir
        self.model_type = model_type
        self.label = label
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fasttext = fasttext_embedding
        if model_type == 'bert':
            self.fasttext = None
    

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = UIT_VSFC(data_dir=os.path.join(self.root_data_dir, "train"), label=self.label, model_type=self.model_type, fasttext_embedding=self.fasttext)
            self.val_data = UIT_VSFC(data_dir=os.path.join(self.root_data_dir, "dev"), label=self.label, model_type=self.model_type, fasttext_embedding=self.fasttext)
        
        if stage == "test":
            self.test_data = UIT_VSFC(data_dir=os.path.join(self.root_data_dir, "test"), label=self.label, model_type=self.model_type, fasttext_embedding=self.fasttext)
    

    def train_dataloader(self):
        if self.model_type == "bert":
            return DataLoader(dataset=self.train_data, batch_size=self.batch_size, collate_fn=bert_collate_fn, num_workers=self.NUM_WORKERS)
        
        if self.model_type == "lstm":
            return DataLoader(dataset=self.train_data, batch_size=self.batch_size, collate_fn=lstm_collate_fn, num_workers=self.NUM_WORKERS)
        

    def val_dataloader(self):
        if self.model_type == "bert":
            return DataLoader(dataset=self.val_data, batch_size=self.batch_size, collate_fn=bert_collate_fn, num_workers=self.NUM_WORKERS)
        
        if self.model_type == "lstm":
            return DataLoader(dataset=self.val_data, batch_size=self.batch_size, collate_fn=lstm_collate_fn, num_workers=self.NUM_WORKERS)
    

    def test_dataloader(self):
        if self.model_type == "bert":
            return DataLoader(dataset=self.test_data, batch_size=self.batch_size, collate_fn=bert_collate_fn, num_workers=self.NUM_WORKERS)
        
        if self.model_type == "lstm":
            return DataLoader(dataset=self.test_data, batch_size=self.batch_size, collate_fn=lstm_collate_fn, num_workers=self.NUM_WORKERS)
