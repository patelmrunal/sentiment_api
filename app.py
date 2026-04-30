import streamlit as st
import torch
import torch.nn.functional as F
import sys
import os

# Add model folder to path so we can import our classes
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))


# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🎬",
    layout="centered"
)

# ── Load model once using Streamlit cache ──────────────────
# @st.cache_resource means this runs only ONE time
# even if the user clicks the button 100 times
# without this, model reloads on every interaction — very slow
@st.cache_resource
def load_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Path must be added INSIDE this function too
    # not just at the top of the file
    # because @st.cache_resource runs in its own context
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

    from dataset import Vocabulary
    from network import SentimentModel

    loaded_word2idx = Vocabulary.load("saved/vocab.pkl")
    vocab = Vocabulary()
    vocab.word2idx = loaded_word2idx

    model = SentimentModel(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(
        torch.load("saved/model.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    return model, vocab, DEVICE

def predict(text, model, vocab, device):
    ids = vocab.encode(text, max_len=200)
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()

    positive = probs[1].item()
    negative = probs[0].item()
    label    = "POSITIVE" if positive > negative else "NEGATIVE"

    return label, positive, negative

# ── UI ─────────────────────────────────────────────────────
st.title("🎬 Sentiment Analyser")
st.markdown("Built with PyTorch + FastAPI — trained on 50,000 IMDB reviews")
st.divider()

# Load model (cached after first run)
model, vocab, device = load_model()

# Text input
review = st.text_area(
    label="Paste a product or movie review below",
    placeholder="This movie was absolutely fantastic...",
    height=150
)

# Analyse button
if st.button("Analyse Sentiment", type="primary"):

    # Guard against empty input
    if not review.strip():
        st.warning("Please enter some text first")

    else:
        label, pos_score, neg_score = predict(review, model, vocab, device)

        st.divider()

        # Show result with colour
        if label == "POSITIVE":
            st.success(f"✅  {label}")
        else:
            st.error(f"❌  {label}")

        # Confidence score
        confidence = max(pos_score, neg_score) * 100
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

        # Progress bars — visual breakdown
        st.markdown("**Score breakdown**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("🟢 Positive")
            st.progress(pos_score)
            st.caption(f"{pos_score * 100:.1f}%")

        with col2:
            st.markdown("🔴 Negative")
            st.progress(neg_score)
            st.caption(f"{neg_score * 100:.1f}%")

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption("Model: Bidirectional LSTM | Accuracy: 85.3% | Dataset: IMDB 50k")