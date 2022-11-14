import config
from trainer import FastTextLSTMModel
from dataset import UIT_VFSC_Dataset, collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
        train_dataset = UIT_VFSC_Dataset(root_dir=config.TRAIN_PATH)
        test_dataset = UIT_VFSC_Dataset(root_dir=config.VALIDATION_PATH)

        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=config.NUM_WORKERS)
        eval_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=config.NUM_WORKERS)

        seed_everything(42)

        system = FastTextLSTMModel()

        checkpoint_callback = ModelCheckpoint(dirpath=config.CHECKPOINT_DIR, monitor="val_loss",
                                                save_top_k=3, mode="min")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", check_finite=True)

        logger = TensorBoardLogger(save_dir=config.TENSORBOARD["DIR"], name=config.TENSORBOARD["NAME"], version=config.TENSORBOARD["VERSION"])

        trainer = Trainer(accelerator=config.ACCELERATOR, check_val_every_n_epoch=config.EVAL_EVERY_EPOCHS,
                        gradient_clip_val=1.0,max_epochs=config.NUM_EPOCHS,
                        enable_checkpointing=True, deterministic=True, default_root_dir=config.CHECKPOINT_DIR,
                        callbacks=[checkpoint_callback, early_stopping], logger=logger)
        trainer.fit(model=system, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=eval_dataloader,
                ckpt_path=config.CONTINUE_TRAINING)

        print("\nTRAINING FINISH...")