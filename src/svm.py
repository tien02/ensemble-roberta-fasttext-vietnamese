import os
import re
import yaml
import joblib
import numpy as np
import pandas as pd

from pyvi import ViTokenizer
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


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


def evaluate(y_true, y_pred, model):
    print(f"Accuracy:{metrics.accuracy_score(y_true=y_true, y_pred=y_pred):.4f} ")
    print(f"Precision: {metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted'):.4f}")
    print(f"Recall: {metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted'):.4f}")
    print(f"F1 score: {metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'):.4f}")

    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=model.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()


def read_data(dir:str, label:str):
    assert label in ["sentiments", "topics"], f"Expect 'label' argument to be 'sentiments' or 'topics', unknown '{label}'."

    sents = pd.read_table(os.path.join(dir,"sents.txt"), names=["sents"])

    if label == "sentiments":
        labels = pd.read_table(os.path.join(dir, "sentiments.txt"), names=["labels"])
    if label == "topics":
        labels = pd.read_table(os.path.join(dir, "topics.txt"), names=["labels"])

    return pd.concat([sents, labels], axis=1)


if __name__ == "__main__":
    with open("./src/config/data.yaml") as f:
        data_config = yaml.safe_load(f)

    with open("./src/config/svm.yaml") as f:
        svm_config = yaml.safe_load(f)
        
    w2v = KeyedVectors.load(svm_config['fasttext_embedding_path'])
    converter = Word2Vec(svm_config['max_length'], w2v.wv)
    # Data

    train_df = read_data(dir=data_config['path']['train'], label=data_config['label'])
    dev_df = read_data(dir=data_config['path']['val'], label=data_config['label'])
    test_df = read_data(dir=data_config['path']['test'], label=data_config['label'])
    train_df = pd.concat([train_df, dev_df], axis=0)

    print("\tData")
    X_train = np.array(train_df["sents"])
    y_train = np.array(train_df['labels'])
    X_test = np.array(test_df["sents"])
    y_test = np.array(test_df['labels'])

    X_train_vec = np.array([converter.text_to_vector(text) for text in X_train])
    X_test_vec = np.array([converter.text_to_vector(text) for text in X_test])

    print(f"X_train shape: {X_train_vec.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_vec.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Model
    fast_svm = Pipeline(steps=[
        ('pca', PCA(n_components=svm_config['pca_components'])),
        ('svm', SVC(probability=True, random_state=42, class_weight="balanced"))
    ], verbose=True)

    print("\n\tModel")
    print(fast_svm)

    # Train model
    print("\n\tTraining")
    fast_svm.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = fast_svm.predict(X_test_vec)
    print("\n\tEvaluation")
    evaluate(y_test, y_pred, fast_svm)

    if not os.path.exists("ssrc/vm_ckpt"):
        os.mkdir("src/svm_ckpt")
    # Save
    joblib.dump(fast_svm, f"src/svm_ckpt/fast_svm.pkl")