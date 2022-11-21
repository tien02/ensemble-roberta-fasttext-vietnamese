from .config import config
import torch
import torch.nn as nn
from .dataset import UIT_VFSC_Dataset
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from pytorch_lightning import LightningModule

dataset = UIT_VFSC_Dataset(root_dir=config.TRAIN_PATH)

class PhoBERTModel(LightningModule):
    def __init__(self, model, use_loss_weight=True):
        super(PhoBERTModel, self).__init__()
        self.model = model
        self.train_loss_fn = nn.CrossEntropyLoss(weight=dataset.class_weights)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics =  MetricCollection([
          Precision(average="weighted", num_classes=config.NUM_CLASSES),
          Recall(average="weighted", num_classes=config.NUM_CLASSES),
          F1Score(average="weighted", num_classes=config.NUM_CLASSES)])
        self.val_acc_fn = Accuracy(average="weighted", num_classes=config.NUM_CLASSES)
        self.use_loss_weight = use_loss_weight

    def forward(self, input_ids, attn_mask):
        return self.model(input_ids, attn_mask)
    
    def training_step(self, batch):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)

        if self.use_loss_weight:
          loss = self.train_loss_fn(pred, label)
        else:
          loss = self.loss_fn(pred, label)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)
        loss = self.loss_fn(pred, label)

        pred = torch.argmax(pred, dim=1)

        metrics = self.test_metrics(pred, label)
        self.log("test_loss", loss)
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, label = batch
    
        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)
        
        loss = self.loss_fn(pred, label)
        self.val_acc_fn.update(pred, label)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        # loss = torch.stack(outputs).mean()
        val_acc = self.val_acc_fn.compute()
        # self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=config.LEARNING_RATE, eps=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE,
                    steps_per_epoch=len(dataset), epochs=config.EPOCHS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }