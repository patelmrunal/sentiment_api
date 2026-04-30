import re
import pickle
import torch
from torch.utils.data import Dataset
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>" : 1}

    def clean(self,text):
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text.strip()
    
    def build(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.clean(text).split()
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)
        
        print(f"Vocabulary size: {len(self.word2idx)}")

    
    def encode(self, text, max_len=200):
        tokens = self.clean(text).split()
        tokens = tokens[:max_len]
        ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        padding = [self.word2idx["<PAD>"]] * (max_len - len(ids))
        ids += padding
        return ids
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.word2idx, f)
        print(f"Vocabulary saved to {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class SentimentDataset(Dataset):
    def __init__(self, df, vocab, max_len=200):
        self.vocab = vocab
        self.max_len = max_len

        self.reviews = df["review"].tolist()
        self.labels = df["sentiment"].map({"positive": 1, "negative": 0}).tolist()
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        ids = self.vocab.encode(review, self.max_len)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return ids_tensor, label_tensor