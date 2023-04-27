import yaml
import torch
import torch.nn as nn
from ..dataset import UIT_VSFC
from torchmetrics import F1Score, Precision, Recall, MetricCollection, Accuracy
from pytorch_lightning import LightningModule

with open("./src/config/data.yaml") as f:
    data_config = yaml.safe_load(f)

dataset = UIT_VSFC(data_dir=data_config['path']['train'], label=data_config['label'],model_type='bert')

class PhoBERTModel(LightningModule):
    def __init__(self, model):
        super(PhoBERTModel, self).__init__()
        self.model = model
        self.train_loss_fn = nn.CrossEntropyLoss(weight=dataset.class_weights)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics =  MetricCollection([
          Precision(average="weighted", num_classes=data_config['out_channels']),
          Recall(average="weighted", num_classes=data_config['out_channels']),
          F1Score(average="weighted", num_classes=data_config['out_channels'])])
        self.val_acc_fn = Accuracy(average="weighted", num_classes=data_config['out_channels'])

    def forward(self, input_ids, attn_mask):
        return self.model(input_ids, attn_mask)
    
    def training_step(self, batch):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)

        loss = self.train_loss_fn(pred, label)

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