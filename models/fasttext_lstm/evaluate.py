import config
from trainer import FastTextLSTMModel
from dataset import UIT_VFSC_Dataset, collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

if __name__ == "__main__":
        test_dataset = UIT_VFSC_Dataset(root_dir=config.VALIDATION_PATH)

        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=config.NUM_WORKERS)

        system = FastTextLSTMModel()

        trainer = Trainer(accelerator=config.ACCELERATOR)

        trainer.test(model=system, ckpt_path=config.TEST_CKPT_PATH, dataloaders=test_dataloader)

        print("\nEVALUATE DONE...")