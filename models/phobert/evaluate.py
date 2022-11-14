import config 
import torch
from torch.utils.data import DataLoader
from dataset import tokenizer, UIT_VFSC_Dataset, collate_fn
from pytorch_lightning import Trainer, seed_everything
from model import PhoBERTLSTM_base, PhoBERTLSTM_large, PhoBertFeedForward_base, PhoBertFeedForward_large
from trainer import PhoBERTModel
from termcolor import colored

def load_model():
    if config.MODEL == "FeedForward-base":
        model = PhoBertFeedForward_base(from_pretrained=False)
        print(colored("\nEvaluate PhoBERT FeedForward base\n", "green"))
    elif config.MODEL == "FeedForward-large":
        model = PhoBertFeedForward_large(from_pretrained=False)
        print(colored("\nEvaluate PhoBERT FeedForward large\n", "green"))
    elif config.MODEL == "LSTM-base":
        model = PhoBERTLSTM_base(from_pretrained=False)
        print(colored("\nEvaluate PhoBERT LSTM base\n", "green"))
    else:
        model = PhoBERTLSTM_large(from_pretrained=False)
        print(colored("\nEvaluate PhoBERT LSTM large\n", "green"))
    system = PhoBERTModel(model)
    return system

if __name__ == "__main__":
    seed_everything(config.SEED)

    system = load_model()

    test_data = UIT_VFSC_Dataset(root_dir=config.TEST_PATH,
                                label=config.LABEL)
    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=config.BATCH_SIZE,
                                collate_fn=collate_fn,
                                shuffle=False,
                                num_workers=config.NUM_WORKERS)

    trainer = Trainer(accelerator=config.ACCELERATOR)

    print(colored("\nEvaluate on Test Set:\n", "green"))
    trainer.test(model=system, ckpt_path=config.TEST_CKPT_PATH, dataloaders=test_dataloader)