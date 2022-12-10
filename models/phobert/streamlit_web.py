import torch
import pandas as pd
from config import config
import streamlit as st
from pyvi import ViTokenizer
from dataset import tokenizer
from trainer import PhoBERTModel
from model import PhoBertFeedForward_base

st.set_page_config(
    page_title="Vietnamese Student Feedback",
    page_icon="🤖",
    layout="wide"
)

@st.cache(allow_output_mutation=True)
def load_model():
    model = PhoBertFeedForward_base(from_pretrained=False)
    system = PhoBERTModel(model)
    checkpoint = torch.load(config.TEST_CKPT_PATH, map_location=device)
    system.load_state_dict(checkpoint["state_dict"])
    system.eval()
    system.freeze()
    return system

map_dict = {
    0: "Negative",
    1: "Neural",
    2: "Positive"
}

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    system = load_model()
    system.to(device)
    st.title("Vietnamese Student Feedback 🤖")
    sentence = st.text_input("Nhập feedback:")
    if sentence:
        sentence = sentence.strip()
        seg_sentence = ViTokenizer.tokenize(sentence.lower())
        tokens = tokenizer(seg_sentence, return_tensors='pt')
        with torch.no_grad():
            output = system(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
            pred_label = int(torch.argmax(output, dim=1).item())
            accuracy = round(torch.max(torch.nn.functional.softmax(output, dim=1)).item() * 100, 3)

            sent, acc = st.columns(2)
            with sent:
                if pred_label == 2:
                    st.markdown("### Possitive 🤗")
                elif pred_label == 1:
                    st.markdown("### Neural 😕")
                else:
                    st.markdown("### Negative 😥")
            with acc:
                st.metric("Probability", str(accuracy) + " %")
                
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Evaluate on Test Set")
        data = [["PhoBERT (base) + FeedForward",	0.92502,	0.92988,	0.92348], 
                ["PhoBERT (large) + FeedForward",	0.91447,	0.90935,	0.88475], 
                ["PhoBERT (base) + LSTM",	0.92399,	0.92893,	0.92259], 
                ["PhoBERT (large) + LSTM",	0.91062,	0.90556,	0.88104], 
                ["FastText + LSTM",	0.84022,	0.86323,	0.84127], 
                ["FastText + SVM",	0.84825,	0.86639,	0.85023]]
        eva_on_test = pd.DataFrame(data=data, columns=["Model", "Precision", "Recall", "F1-score"])
        st.table(eva_on_test)

        st.markdown("## Evaluate on Test Set with class weight")
        data = [["PhoBERT (base) + FeedForward",	0.92867,	0.92672,	0.92751], 
                ["PhoBERT (large) + FeedForward",	0.90756,	0.9024,	0.87796], 
                ["PhoBERT (base) + LSTM",	0.92489,	0.92356,	0.92407], 
                ["PhoBERT (large) + LSTM",	0.90965,	0.90461,	0.8801], 
                ["FastText + LSTM",	0.85727,	0.81207,	0.83015], 
                ["FastText + SVM",	0.85376,	0.86229,	0.85561]]
        eva_on_test = pd.DataFrame(data=data, columns=["Model", "Precision", "Recall", "F1-score"])
        st.table(eva_on_test)
    
    with col2:
        st.markdown("## Emsemble evaluation on Test Set")
        data = [["(2) + (6)",	0.5,	0.89417,	0.91124,	0.88877], 
                ["(2) + (4)",	0.7,	0.91587,	0.91093,	0.88627], 
                ["(2) + (5)",	0.8,	0.91521,	0.91030,	0.88565], 
                ["(4) + (6)",	0.2,	0.89082,	0.90556,	0.88562], 
                ["(4) + (5)",	0.7,	0.91145,	0.90651,	0.88195], 
                ["(5) + (6)",	0.4,	0.85532,	0.87208,	0.85340]]
        eva_on_test = pd.DataFrame(data=data, columns=["Model", "Ratio", "Precision", "Recall", "F1-score"])
        st.table(eva_on_test)

        st.markdown("## Ensemble Evaluation on Test set with class weights")
        data = [["(1) + (4)",	0.8,	0.92845,	0.92956,	0.92889], 
                ["(1) + (2)",	0.9,	0.92899,	0.92798,	0.92837], 
                ["(1) + (6)",	0.5,	0.92932,	0.92830,	0.92830], 
                ["(1) + (5)",	0.9,	0.92943,	0.92672,	0.92783], 
                ["(3) + (4)",	0.8,	0.92507,	0.92704,	0.92584], 
                ["(3) + (6)",	0.8,	0.92545,	0.92451,	0.92484],
                ["(3) + (5)",	0.6,	0.92654,	0.92356,	0.92474]]
        eva_on_test = pd.DataFrame(data=data, columns=["Model", "Ratio", "Precision", "Recall", "F1-score"])
        st.table(eva_on_test)
