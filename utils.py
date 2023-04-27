import yaml

from src.models import *
from src.dataset import DataModule
from src.trainers import PhoBERTModel, FastTextLSTMModel

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def load_data(path_to_yaml_file:str):
    with open(path_to_yaml_file) as f:
        data_config = yaml.safe_load(f)
    dm = DataModule(root_data_dir=data_config['root_data_dir'], 
                    label=data_config['label'], 
                    model_type=data_config['model_type'], 
                    batch_size=data_config['batch_size'], 
                    num_workers=data_config['num_workers'], 
                    fasttext_embedding=data_config['fasttext_embedding'])
    return dm
    

def load_model(path_to_yaml_file:str):
    with open(path_to_yaml_file) as f:
        model_config = yaml.safe_load(f)

    name = model_config['model_name']
    # model configuration
    if name == "BERT-FF-BASE":
        model = PhoBertFeedForward_base(from_pretrained=model_config['from_pretrained'],
                                        freeze_backbone=model_config['freeze_backbone'],
                                        drop_out=model_config['drop_out'],
                                        out_channels=model_config['out_channels'])
    elif name == "BERT-FF-LARGE":
        model = PhoBertFeedForward_large(from_pretrained=model_config['from_pretrained'],
                                        freeze_backbone=model_config['freeze_backbone'],
                                        drop_out=model_config['drop_out'],
                                        out_channels=model_config['out_channels'])
    elif name == "BERT-LSTM-BASE":
        model = PhoBERTLSTM_base(from_pretrained=model_config['from_pretrained'],
                                        freeze_backbone=model_config['freeze_backbone'],
                                        drop_out=model_config['drop_out'],
                                        out_channels=model_config['out_channels'])
    elif name == "BERT-LSTM-LARGE":
        model = PhoBERTLSTM_large(from_pretrained=model_config['from_pretrained'],
                                        freeze_backbone=model_config['freeze_backbone'],
                                        drop_out=model_config['drop_out'],
                                        out_channels=model_config['out_channels'])
    elif name == "FASTTEXT-LSTM":
        pass
    else:
        raise ValueError(f"Not support {name}")
    
    # system configuration
    if name.startswith("FASTTEXT"):
        system = FastTextLSTMModel(dropout=model_config['drop_out'])
    else:
        system = PhoBERTModel(model=model)
    
    return system


def load_trainer(path_to_yaml_file:str):
    with open(path_to_yaml_file) as f:
        trainer_config = yaml.safe_load(f)

    checkpoint_callback = ModelCheckpoint(dirpath=trainer_config['ckpt_dir'], monitor="val_loss", save_top_k=3, mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = TensorBoardLogger(save_dir=trainer_config['tensorboard']['dir'], name=trainer_config['tensorboard']['name'], version=trainer_config['tensorboard']['version'])

    trainer = Trainer(accelerator=trainer_config['accelarator'], check_val_every_n_epoch=trainer_config['val_each_epoch'],
                    gradient_clip_val=1.0,max_epochs=trainer_config['max_epochs'],
                    enable_checkpointing=True, deterministic=True, default_root_dir=trainer_config['ckpt_dir'],
                    callbacks=[checkpoint_callback, early_stopping], logger=logger, accumulate_grad_batches=4,log_every_n_steps=100)

    return trainer