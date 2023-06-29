import os
import yaml
import argparse
from lightning.pytorch import seed_everything
from utils import load_data, load_model, load_trainer

def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, action='store', choices=['train', 'test'], default='train', help="Perform the choosen task, 'train' for training, 'test' for evaluate on test set")
        parser.add_argument('--data', type=str, action='store', default='./config/data.yaml', help="Path to DATA configuration .yaml")
        parser.add_argument('--model', type=str, action='store', default='./config/model.yaml', help="Path to MODEL configuration .yaml")
        parser.add_argument('--trainer', type=str, action='store', default='./config/trainer.yaml', help="Path to TRAINER configuration .yaml")
        opt = parser.parse_args()
        return opt

def run(task:str, data:str, model:str, trainer:str):
        with open(trainer) as f:
                trainer_config = yaml.safe_load(f)

        seed_everything(trainer_config['seed'])

        trainer_module = load_trainer(trainer)
        
        dm = load_data(data)

        if task == "train":
                dm.setup(stage='fit')
                model_module = load_model(model,loss_weight=dm.train_data.class_weights)
                
                trainer_module.fit(model=model_module, datamodule=dm, ckpt_path=trainer_config['keep_training_path'])
        if task == "test":
                dm.setup(stage='test')
                model_module = load_model(model)
                if trainer_config['test_ckpt'] is None:
                        print("Please specify the test_ckpt")
                        return
                trainer_module.test(model=model_module, datamodule=dm, ckpt_path=trainer_config['test_ckpt'])

def main(opt):
       run(**vars(opt))

if __name__ == "__main__":
        opt = parse_opt()
        main(opt)