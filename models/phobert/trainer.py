import config
import torch
import torch.nn as nn
from .dataset import UIT_VFSC_Dataset
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall
from pytorch_lightning import LightningModule

dataset = UIT_VFSC_Dataset(root_dir=config.TRAIN_PATH)

class PhoBERTModel(LightningModule):
    def __init__(self, model):
        super(PhoBERTModel, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(average="weighted", num_classes=config.NUM_CLASSES)
        self.precision_fn = Precision(average="weighted", num_classes=config.NUM_CLASSES)
        self.recall_fn = Recall(average="weighted", num_classes=config.NUM_CLASSES)
        self.f1 = F1Score(average="weighted", num_classes=config.NUM_CLASSES)


    def forward(self, input_ids, attn_mask):
        return self.model(input_ids, attn_mask)
    
    def training_step(self, batch):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, label)

        acc = self.acc(pred, label)
        pre = self.precision_fn(pred, label)
        re = self.recall_fn(pred, label)
        f1 = self.f1(pred, label)
        
        self.log_dict({
            "test_loss": loss,
            "test_accuracy": acc,
            "test_f1": f1,
            "test_precision": pre,
            "test_recall": re 
        }, on_epoch=True, prog_bar=True)
        

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, label = batch
    
        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)
        loss = self.loss_fn(pred, label)
        acc = self.acc(pred, label)

        self.log_dict({
            "val_loss": loss,
            "val_accuracy": acc,
        }, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=config.LEARNING_RATE, eps=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE,
                    steps_per_epoch=len(dataset), epochs=config.EPOCHS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }