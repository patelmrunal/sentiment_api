import torch
import torch.nn.functional as F
import sys
import os
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from model.network import SentimentModel
from model.dataset import Vocabulary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_DIR = os.path.normpath(os.path.join(BASE_DIR, "../saved"))
VOCAB_PATH = os.path.join(SAVED_DIR, "vocab.pkl")
MODEL_PATH = os.path.join(SAVED_DIR, "model.pt")

class SentimentPredictor:
    def __init__(self):
        # Load vocabulary — must be the exact same one used during training
        loaded_word2idx = Vocabulary.load(VOCAB_PATH)
        self.vocab = Vocabulary()
        self.vocab.word2idx = loaded_word2idx

        # Rebuild model architecture — must match train.py exactly
        self.model = SentimentModel(vocab_size=len(self.vocab)).to(DEVICE)

        # Load the trained weights into the architecture
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )

        # Set to eval mode — turns off dropout for inference
        self.model.eval()
        print("Model loaded and ready")

    def predict(self, text: str) -> dict:
        # Encode the raw text exactly as done during training
        ids = self.vocab.encode(text, max_len=200)

        # Add batch dimension — model expects [batch, seq] not just [seq]
        # unsqueeze(0) turns shape [200] into [1, 200]
        tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = self.model(tensor)                    # [1, 2]
            probs  = F.softmax(logits, dim=1).squeeze()   # [2]

        negative_score = probs[0].item()
        positive_score = probs[1].item()
        predicted_label = "positive" if positive_score > negative_score else "negative"

        return {
                "label":      predicted_label,
                "confidence": round(max(positive_score, negative_score), 4),
                "scores": {
                    "positive": round(positive_score, 4),
                    "negative": round(negative_score, 4)
                }
            }


if __name__ == "__main__":
    predictor = SentimentPredictor()

    tests = [
        "This movie was absolutely fantastic, I loved every minute",
        "Terrible film, complete waste of time, worst movie ever",
        "It was okay, nothing special but not bad either",
        "The acting was great but i don't like story but last part was good also movie was good but something that i didn't like was middle part"
    ]

    for text in tests:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Result: {result}")