import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import multiprocessing as mp
import sys
sys.path.append("../model")

from dataset import Vocabulary, SentimentDataset
from network import SentimentModel

# ── Configuration ──────────────────────────────────────────
# Change these numbers in one place, affects everything below
BATCH_SIZE = 64      # how many reviews per training step
EPOCHS     = 5       # how many times to loop through all data
LR         = 1e-3    # learning rate — how big each weight update is
MAX_LEN    = 200
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# ── Helper Functions ───────────────────────────────────────
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()   # tells dropout to activate — important
    total_loss = 0
    correct    = 0

    for batch_idx, (reviews, labels) in enumerate(loader):
        reviews = reviews.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        # Forward pass
        predictions = model(reviews)

        # Calculate how wrong we are
        loss = loss_fn(predictions, labels)

        # Backward pass — figure out which weights caused the error
        optimizer.zero_grad()   # clear gradients from last step
        loss.backward()         # calculate new gradients
        optimizer.step()        # update weights

        total_loss += loss.item()
        correct    += (predictions.argmax(1) == labels).sum().item()

        # Print progress every 100 batches so you can see it learning
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader, loss_fn, device):
    model.eval()    # tells dropout to deactivate — important
    total_loss = 0
    correct    = 0

    with torch.inference_mode():   # no gradient tracking needed
        for reviews, labels in loader:
            reviews = reviews.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            predictions = model(reviews)
            loss        = loss_fn(predictions, labels)

            total_loss += loss.item()
            correct    += (predictions.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def main():
    print(f"Training on: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA unavailable in current torch build/environment.")

    # ── Load Data ──────────────────────────────────────────────
    df = pd.read_csv("../Data/Raw/IMDB-Dataset.csv")
    vocab = Vocabulary(min_freq=2)
    vocab.build(df["review"].tolist())
    vocab.save("../saved/vocab.pkl")

    dataset = SentimentDataset(df, vocab, MAX_LEN)
    train_size = int(0.8 * len(dataset))   # 40,000 reviews for training
    val_size = len(dataset) - train_size   # 10,000 reviews for validation

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # DataLoader feeds batches to the model automatically
    # shuffle=True means every epoch sees reviews in different order
    # this prevents the model memorising the order instead of the content
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # ── Model, Loss, Optimizer ─────────────────────────────────
    model = SentimentModel(vocab_size=len(vocab)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training Loop ──────────────────────────────────────────
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, DEVICE
        )

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save the best model — not the last one
        # The last epoch is not always the best one
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), "../saved/model.pt")
            print(f"Model saved — best val accuracy: {best_val_accuracy:.4f}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()