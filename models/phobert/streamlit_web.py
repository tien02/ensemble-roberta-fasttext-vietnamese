import torch
import streamlit as st
from pyvi import ViTokenizer
from dataset import tokenizer
from trainer import PhoBERTModel
from model import PhoBertFeedForward_base

@st.cache(allow_output_mutation=True)
def load_model():
    CKPT_PATH =  "checkpoint/epoch=23-step=4296.ckpt"
    model = PhoBertFeedForward_base(from_pretrained=False)
    system = PhoBERTModel(model)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
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

            if pred_label == 2:
                st.markdown("### Possitive 🤗")
            elif pred_label == 1:
                st.markdown("### Neural 😕")
            else:
                st.markdown("### Negative 😥")
            
            st.metric("Probability", str(accuracy) + " %")
    else:
        st.error("You should type something first! 😵")
