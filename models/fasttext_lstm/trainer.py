from .config import config
import torch
import torch.nn as nn
from .dataset import UIT_VFSC_Dataset
from .model import FastTextLSTM
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall
from pytorch_lightning import LightningModule

dataset = UIT_VFSC_Dataset(config.TRAIN_PATH, label=config.LABEL)

class FastTextLSTMModel(LightningModule):
    def __init__(self):
        super(FastTextLSTMModel, self).__init__()
        self.model = FastTextLSTM(config.VECTOR_SIZE, config.OUT_CHANNELS)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(average="weighted", num_classes=config.OUT_CHANNELS)
        self.f1 = F1Score(average="weighted", num_classes=config.OUT_CHANNELS)
        self.precision_fn = Precision(average="weighted", num_classes=config.OUT_CHANNELS)
        self.recall_fn = Recall(average="weighted", num_classes=config.OUT_CHANNELS)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, batch_size=config.BATCH_SIZE, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, y)

        acc = self.acc(pred, y)
        f1 = self.f1(pred, y)
        pre = self.precision_fn(pred, y)
        re = self.recall_fn(pred, y)

        self.log_dict({
            "test_loss": loss,
            "test_accuracy": acc,
            "test_f1": f1,
            "test_precision": pre,
            "test_recall": re
        }, on_epoch=True, batch_size=config.BATCH_SIZE, prog_bar=True)
        

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, y)

        acc = self.acc(pred, y)


        self.log_dict({
            "val_loss": loss,
            "val_accuracy": acc,}, 
            on_epoch=True, batch_size=config.BATCH_SIZE, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=config.LEARNING_RATE, eps=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE,
                    steps_per_epoch=len(dataset), epochs=config.NUM_EPOCHS)
        return {
            "optimizer":optimizer,
            "lr_scheduler": scheduler
        }