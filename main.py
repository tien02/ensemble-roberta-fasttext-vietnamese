from src.models import *
from src.dataset import DataModule
from src.trainers import FastTextLSTMModel, PhoBERTModel
from lightning.pytorch.cli import LightningCLI

def cli_main():
    cli = LightningCLI(model_class=PhoBERTModel, datamodule_class=DataModule)


if __name__ == "__main__":
    cli_main()