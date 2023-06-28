import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from .data import TextDataset
from .utils import bert_collate_fn, lstm_collate_fn

class DataModule(LightningDataModule):
    def __init__(self, root_data_dir:str, model_type:str, batch_size:int, num_workers:int, fasttext_embedding:str=None):
        super().__init__()
        self.root_data_dir = root_data_dir
        if model_type == "bert":
            self.collate_fn = bert_collate_fn
        else:
            self.collate_fn = lstm_collate_fn

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fasttext = fasttext_embedding
        if model_type == 'bert':
            self.fasttext = None
        self.model_type = model_type
    

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = TextDataset(data_dir=os.path.join(self.root_data_dir, "train"), model_type=self.model_type, fasttext_embedding=self.fasttext)
            self.val_data = TextDataset(data_dir=os.path.join(self.root_data_dir, "dev"), model_type=self.model_type, fasttext_embedding=self.fasttext)
        
        if stage == "test":
            self.test_data = TextDataset(data_dir=os.path.join(self.root_data_dir, "test"), model_type=self.model_type, fasttext_embedding=self.fasttext)
    

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
        

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)


    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)