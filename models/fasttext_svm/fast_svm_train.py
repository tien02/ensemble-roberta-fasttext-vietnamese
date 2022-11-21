from .config import config
import joblib
import numpy as np
import pandas as pd
from .utils import Word2Vec, evaluate

from gensim.models import KeyedVectors

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

w2v = KeyedVectors.load(config.EMBEDDING_PATH)
converter = Word2Vec(config.MAX_LENGTH, w2v.wv)

if __name__ == "__main__":
    # Data

    train_df = pd.read_csv(config.TRAIN_PATH, sep="\t")
    dev_df = pd.read_csv(config.VALIDATION_PATH, sep="\t")
    test_df = pd.read_csv(config.TEST_PATH, sep="\t")
    train_df = pd.concat([train_df, dev_df], axis=0)

    print("\tData")
    X_train = np.array(train_df["sents"])
    y_train = np.array(train_df[config.LABEL])
    X_test = np.array(test_df["sents"])
    y_test = np.array(test_df[config.LABEL])

    X_train_vec = np.array([converter.text_to_vector(text) for text in X_train])
    X_test_vec = np.array([converter.text_to_vector(text) for text in X_test])

    print(f"X_train shape: {X_train_vec.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_vec.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Model
    fast_svm = Pipeline(steps=[
        ('pca', PCA(n_components=config.PCA_COMPONENTS)),
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

    # Save
    joblib.dump(fast_svm, config.SAVE_PATH)