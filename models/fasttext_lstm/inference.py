import config
import torch
import torch.nn as nn
from trainer import FastTextLSTMModel
from termcolor import colored
from utils import preprocess_fn
from gensim.models import KeyedVectors

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
system = FastTextLSTMModel()
system.to(device)
checkpoint = torch.load(config.TEST_CKPT_PATH, map_location=device)
system.load_state_dict(checkpoint["state_dict"])
system.eval()
system.freeze()

map_dict = {
    0: "Negative",
    1: "Neural",
    2: "Positive"
}

word_vec = KeyedVectors.load(config.FAST_TEXT_PRETRAINED)

if __name__ == "__main__":
    print("Enter -1 to exit...")
    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "-1":    break
        tokens = preprocess_fn(sentence)
        x_embed = []
        for token in tokens:
            try:
                x_embed.append(torch.unsqueeze(torch.tensor(word_vec.wv[token]), dim=0))
            except Exception as e:
                pass
        x_embed = torch.unsqueeze(torch.cat(x_embed), dim=1).to(device)
        print(x_embed.size())
        with torch.no_grad():
            logits = system(x_embed)
            pred = nn.functional.softmax(logits, dim=1)
            pred_label = int(torch.argmax(pred, dim=1).item())
            accuracy = round(pred[0][pred_label].item(), 3)
            if pred_label == 2:
                print("\tResult: "+ colored(map_dict[pred_label], "blue") + ", Accuracy: " + colored(accuracy, "blue") + "\n")
            elif pred_label == 1:
                print("\tResult: "+ colored(map_dict[pred_label], "yellow") + ", Accuracy: " + colored(accuracy, "yellow") + "\n")
            else:
                print("\tResult: "+ colored(map_dict[pred_label], "red") + ", Accuracy: " + colored(accuracy, "red") + "\n")
    print(colored("Done...", "green"))