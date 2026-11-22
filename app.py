import re
import json
from pathlib import Path

import torch
import torch.nn as nn
import streamlit as st

MAX_LEN = 300
HERE = Path(__file__).parent


class SeqModel(nn.Module):
    def __init__(self, vocab_size, rnn_type='gru', embed_dim=128, hidden_dim=128):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.rnn(emb)
        if self.rnn_type == 'lstm':
            h = h[0]
        return self.fc(self.dropout(h[-1])).squeeze(1)


@st.cache_resource
def load_model():
    with open(HERE / 'vocab.json') as f:
        vocab = json.load(f)
    model = SeqModel(len(vocab), 'gru')
    model.load_state_dict(torch.load(HERE / 'models' / 'gru.pth', map_location='cpu'))
    model.eval()
    return model, vocab


def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r"[^a-z' ]", ' ', text)
    return text.split()


def predict(text, model, vocab):
    tokens = clean_text(text)
    ids = [vocab.get(w, 1) for w in tokens][:MAX_LEN]
    ids = [0] * (MAX_LEN - len(ids)) + ids
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob, len(tokens)


st.title("Movie Review Sentiment Analyzer")
st.caption("A GRU model trained on 50,000 IMDB reviews (88.6% test accuracy).")

model, vocab = load_model()

review = st.text_area("Write or paste a movie review:", height=180)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please type a review first.")
    else:
        prob, n_words = predict(review, model, vocab)
        if n_words < 5:
            st.info("Very short text, so the prediction may be unreliable.")
        if prob >= 0.5:
            st.success(f"Positive ({prob * 100:.1f}% confident)")
        else:
            st.error(f"Negative ({(1 - prob) * 100:.1f}% confident)")
        st.progress(prob, text=f"Positive probability: {prob * 100:.1f}%")

st.caption("Trained on movie reviews only, so it may not work well on other kinds of text.")