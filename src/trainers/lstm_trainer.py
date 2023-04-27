import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from ..dataset import UIT_VSFC
from ..models import BiLSTM

from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from pytorch_lightning import LightningModule

with open("./src/config/data.yaml") as f:
    data_config = yaml.safe_load(f)

with open("src/config/fasttext.yaml") as f:
    fasttext_config = yaml.safe_load(f)

with open("./config/trainer.yaml") as f:
    trainer_config = yaml.safe_load(f)

dataset = UIT_VSFC(data_dir=data_config['path']['train'], label=data_config['label'],model_type='lstm', fasttext_embedding=fasttext_config['fasttext_embedding_path'])

class FastTextLSTMModel(LightningModule):
    def __init__(self, dropout:float = 0.1):
        super(FastTextLSTMModel, self).__init__()
        self.model = BiLSTM(vector_size=fasttext_config['vector_size'],out_channels=data_config['out_channels'], drop_out=dropout)
        self.train_loss_fn = nn.CrossEntropyLoss(weight=dataset.class_weights)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics =  MetricCollection([
          Precision(average="weighted", num_classes=data_config['out_channels']),
          Recall(average="weighted", num_classes=data_config['out_channels']),
          F1Score(average="weighted", num_classes=data_config['out_channels'])])
        self.val_acc_fn = Accuracy(average="weighted", num_classes=data_config['out_channels'])

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.train_loss_fn(pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
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

    def on_validation_epoch_end(self):
        val_acc = self.val_acc_fn.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=trainer_config['learning_rate'], eps=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=trainer_config['learning_rate'],
                    steps_per_epoch=len(dataset), epochs=trainer_config['max_epochs'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }