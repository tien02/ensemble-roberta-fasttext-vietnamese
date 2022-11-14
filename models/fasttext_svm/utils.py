import re
import numpy as np
import matplotlib.pyplot as plt
from pyvi import ViTokenizer
from sklearn import metrics

def preprocess(text):
    sent = re.sub(r'[^\w\s]', '', text)
    tokens = ViTokenizer.tokenize(sent)
    tokens = tokens.split()
    for token in tokens:
        for t in token:
            if t.isnumeric() or t.isdigit():
                tokens.remove(token)
                break
    return tokens

class Word2Vec():
    def __init__(self, seq_len, embedding_matrix):
        self.seq_len = seq_len
        self.embeded_matrix = embedding_matrix

    def text_to_vector(self, text):
        tokens = preprocess(text)
        emb_vec = []
        for token in tokens:
            try:
                emb_vec.append(self.embeded_matrix[token])
            except Exception as e:
                pass
        useless_len = self.seq_len - len(emb_vec)
        if useless_len > 0:
            for _ in range(useless_len):
                emb_vec.append(np.zeros(300, ))
        elif useless_len < 0:
            emb_vec = emb_vec[:self.seq_len]
        else:
            pass
        return np.array(emb_vec).flatten()

def evaluate(y_true, y_pred, model):
    print(f"Accuracy:{metrics.accuracy_score(y_true=y_true, y_pred=y_pred):.4f} ")
    print(f"Precision: {metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted'):.4f}")
    print(f"Recall: {metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted'):.4f}")
    print(f"F1 score: {metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'):.4f}")

    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=model.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()