from .config import config 
from torch.utils.data import DataLoader
from .dataset import UIT_VFSC_Dataset, collate_fn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .trainer import PhoBERTModel
from pytorch_lightning.loggers import TensorBoardLogger
from .model import PhoBERTLSTM_base, PhoBERTLSTM_large, PhoBertFeedForward_base, PhoBertFeedForward_large
from termcolor import colored

def main():
        seed_everything(config.SEED)

        train_data = UIT_VFSC_Dataset(root_dir=config.TRAIN_PATH, label=config.LABEL)
        eval_data = UIT_VFSC_Dataset(root_dir=config.VALIDATION_PATH, label=config.LABEL)

        train_dataloader = DataLoader(dataset=train_data, collate_fn=collate_fn, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        eval_dataloader = DataLoader(dataset=eval_data, collate_fn=collate_fn, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

        
        if config.MODEL == "FeedForward-base":
                model = PhoBertFeedForward_base()
                print(colored("\nUse PhoBERT FeedForward base\n", "green"))
        elif config.MODEL == "FeedForward-large":
                model = PhoBertFeedForward_large()
                print(colored("\nUse PhoBERT FeedForward large\n", "green"))
        elif config.MODEL == "LSTM-base":
                model = PhoBERTLSTM_base()
                print(colored("\nUse PhoBERT LSTM base\n", "green"))
        else:
                model = PhoBERTLSTM_large()
                print(colored("\nUse PhoBERT LSTM large\n", "green"))
        system = PhoBERTModel(model=model)

        checkpoint_callback = ModelCheckpoint(dirpath= config.CHECKPOINT_DIR, monitor="val_loss",
                                                save_top_k=3, mode="min")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min")

        logger = TensorBoardLogger(save_dir=config.TENSORBOARD["DIR"], name=config.TENSORBOARD["NAME"], version=config.TENSORBOARD["VERSION"])

        trainer = Trainer(accelerator=config.ACCELERATOR, check_val_every_n_epoch=config.VAL_EACH_EPOCH,
                        gradient_clip_val=1.0,max_epochs=config.EPOCHS,
                        enable_checkpointing=True, deterministic=True, default_root_dir=config.CHECKPOINT_DIR,
                        callbacks=[checkpoint_callback, early_stopping], logger=logger, accumulate_grad_batches=4)
        
        # trainer = Trainer(accelerator=config.ACCELERATOR, check_val_every_n_epoch=config.VAL_EACH_EPOCH,
        #                 gradient_clip_val=1.0,max_epochs=config.EPOCHS,
        #                 enable_checkpointing=True, deterministic=True, default_root_dir=config.CHECKPOINT_DIR,
        #                 callbacks=[checkpoint_callback], logger=logger)

        trainer.fit(model=system, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=eval_dataloader, 
                ckpt_path=config.CONTINUE_TRAINING)

if __name__ == "__main__":
        main()
        print("TRAINING FINÍH")