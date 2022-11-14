import re
import time
import pandas as pd
import gensim
from tqdm import tqdm
from pyvi import ViTokenizer

if __name__ == "__main__":
    train_df = pd.read_csv("uit_vsfc_data/train.csv", sep="\t")
    dev_df = pd.read_csv("uit_vsfc_data/dev.csv", sep="\t")
    df = pd.concat([train_df, dev_df], axis=0)
    df.shape

    sents = df["sents"]
    sents = list(sents)
    print(f"Total sentences: {len(sents)}")

    x_tokenized = []
    for sent in tqdm(sents, leave=True):
        sent = re.sub(r'[^\w\s]', '', sent)
        tokens = ViTokenizer.tokenize(sent)
        tokens = tokens.split()
        for token in tokens:
            for t in token:
                if t.isnumeric() or t.isdigit():
                    tokens.remove(token)
                    break
        x_tokenized.append(tokens)

    start = time.time()
    model = gensim.models.FastText(sentences=x_tokenized, vector_size=300, window=3, min_count=5, workers=4)
    model.build_vocab(corpus_iterable=x_tokenized)
    model.train(corpus_iterable=x_tokenized, total_examples=len(x_tokenized), epochs=15)
    end = time.time()
    total_time = end - start
    print("Training finish!")
    print(f"FastText training total time is: {total_time:.2f} seconds")
    model.save("fasttext_train_dev.model")
    print(f"Save to 'fasttext_train_dev.model'")