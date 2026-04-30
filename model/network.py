import torch
from torch import nn

class SentimentModel(nn.Module):
  def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=2, dropout=0.5):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, 
                        hidden_dim, 
                        batch_first=True,
                        bidirectional=True)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_dim * 2, num_classes)

  def forward(self, x):
    embedded = self.embedding(x)
    output, (hidden, cell) = self.lstm(embedded)
    hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
    output = self.fc(self.dropout(hidden))
    return output