from .config import config
import torch
import torch.nn as nn
from .dataset import UIT_VFSC_Dataset
from .model import FastTextLSTM
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from pytorch_lightning import LightningModule

dataset = UIT_VFSC_Dataset(config.TRAIN_PATH, label=config.LABEL)

class FastTextLSTMModel(LightningModule):
    def __init__(self):
        super(FastTextLSTMModel, self).__init__()
        self.model = FastTextLSTM(config.VECTOR_SIZE, config.OUT_CHANNELS)
        self.train_loss_fn = nn.CrossEntropyLoss(weight=dataset.class_weights)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics =  MetricCollection([
          Precision(average="weighted", num_classes=config.OUT_CHANNELS),
          Recall(average="weighted", num_classes=config.OUT_CHANNELS),
          F1Score(average="weighted", num_classes=config.OUT_CHANNELS)])
        self.val_acc_fn = Accuracy(average="weighted", num_classes=config.OUT_CHANNELS)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.train_loss_fn(pred, y)
        self.log("train_loss", loss, batch_size=config.BATCH_SIZE, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, y)
        metrics = self.test_metrics(pred, y)
        self.log("test_loss", loss)
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, y)

        self.val_acc_fn.update(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc_fn.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=config.LEARNING_RATE, eps=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE,
                    steps_per_epoch=len(dataset), epochs=config.NUM_EPOCHS)
        return {
            "optimizer":optimizer,
            "lr_scheduler": scheduler
        }