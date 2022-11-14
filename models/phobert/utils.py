import re
from pyvi import ViTokenizer

def preprocess_fn(text):
    sent = re.sub(r'[^\w\s]', '', text)
    tokens = ViTokenizer.tokenize(sent)
    tokens = tokens.split()
    for token in tokens:
        for t in token:
            if t.isnumeric() or t.isdigit():
                tokens.remove(token)
                break
    return " ".join(tokens)