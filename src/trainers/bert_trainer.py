import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torchmetrics import F1Score, Precision, Recall, MetricCollection, Accuracy
from lightning.pytorch import LightningModule

with open("./config/trainer.yaml") as f:
    trainer_config = yaml.safe_load(f)

class PhoBERTModel(LightningModule):
    def __init__(self, model, out_channels:int, loss_weight=None):
        super(PhoBERTModel, self).__init__()
        self.model = model
        self.train_loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics =  MetricCollection([
          Precision(average="weighted", num_classes=out_channels),
          Recall(average="weighted", num_classes=out_channels),
          F1Score(average="weighted", num_classes=out_channels)])
        self.val_acc_fn = Accuracy(average="weighted", num_classes=out_channels)

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
    
    def on_validation_epoch_end(self,):
        # loss = torch.stack(outputs).mean()
        val_acc = self.val_acc_fn.compute()
        # self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=float(trainer_config['learning_rate']), eps=1e-6, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(trainer_config['learning_rate']),
        #             steps_per_epoch=self.trainer.estimated_stepping_batches, epochs=trainer_config['max_epochs'])
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler
        # }
        return optimizer