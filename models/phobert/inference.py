import config
import torch
from pyvi import ViTokenizer
from dataset import tokenizer
from model import PhoBERTLSTM_base, PhoBERTLSTM_large, PhoBertFeedForward_base, PhoBertFeedForward_large
from trainer import PhoBERTModel
from termcolor import colored

if train_config.MODEL == "FeedForward-base":
    model = PhoBertFeedForward_base(from_pretrained=False)
    print(colored("\nUse PhoBERT FeedForward base\n", "green"))
elif train_config.MODEL == "FeedForward-large":
    model = PhoBertFeedForward_large(from_pretrained=False)
    print(colored("\nUse PhoBERT FeedForward large\n", "green"))
elif train_config.MODEL == "LSTM-base":
    model = PhoBERTLSTM_base(from_pretrained=False)
    print(colored("\nUse PhoBERT LSTM base\n", "green"))
else:
    model = PhoBERTLSTM_large(from_pretrained=False)
    print(colored("\nUse PhoBERT LSTM large\n", "green"))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
system = PhoBERTModel(model)
system.to(device)
checkpoint = torch.load(train_config.CKPT_PATH, map_location=device)
system.load_state_dict(checkpoint["state_dict"])
system.eval()
system.freeze()

map_dict = {
    0: "Negative",
    1: "Neural",
    2: "Positive"
}

if __name__ == "__main__":
    print("Enter -1 to exit...")
    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "-1":    break
        seg_sentence = ViTokenizer.tokenize(sentence.lower())
        tokens = tokenizer(seg_sentence, return_tensors='pt')
        with torch.no_grad():
            output = system(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
            pred_label = int(torch.argmax(output, dim=1).item())
            accuracy = round(output[0][pred_label].item(), 3)
            if pred_label == 2:
                print("\tResult: "+ colored(map_dict[pred_label], "blue") + ", Accuracy: " + colored(accuracy, "blue") + "\n")
            elif pred_label == 1:
                print("\tResult: "+ colored(map_dict[pred_label], "yellow") + ", Accuracy: " + colored(accuracy, "yellow") + "\n")
            else:
                print("\tResult: "+ colored(map_dict[pred_label], "red") + ", Accuracy: " + colored(accuracy, "red") + "\n")
    print(colored("Done...", "green"))