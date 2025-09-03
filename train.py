import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import HierarchicalDataset, collate_fn
from src.model import HierarchicalModel

# --- Hyperparameters ---
NUM_EPOCHS = 20  # Increased epochs for better convergence on the toy dataset
LEARNING_RATE = 0.002
EMBEDDING_DIM = 64
SENTENCE_HIDDEN_DIM = 128
DOCUMENT_HIDDEN_DIM = 128
DATA_PATH = 'data/toy_data.json'
# ---

def train():
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("1. Loading data...")
    try:
        dataset = HierarchicalDataset(json_path=DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # As decided, batch size is 1. Our model processes one document at a time.
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    vocab = dataset.vocab
    vocab_size = len(vocab)
    print(f"   Vocabulary size: {vocab_size}")

    print("2. Initializing model...")
    model = HierarchicalModel(
        vocab=vocab,
        emb_dim=EMBEDDING_DIM,
        sent_hidden=SENTENCE_HIDDEN_DIM,
        doc_hidden=DOCUMENT_HIDDEN_DIM
    ).to(device)

    # We use CrossEntropyLoss, but importantly, we ignore the padding token index.
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("3. Starting training...")
    model.train()  # Set the model to training mode

    for epoch in range(NUM_EPOCHS):
        total_epoch_loss = 0

        # The data_loader yields one processed document at a time due to our collate_fn
        for i, doc in enumerate(data_loader):
            # Move all tensors in the document to the selected device
            doc_to_device = [[sent.to(device) for sent in para] for para in doc]
            # Flatten paragraphs into a single list of sentences for the model
            sentences_to_process = [sent for para in doc_to_device for sent in para]

            if not sentences_to_process:
                continue

            optimizer.zero_grad()

            # The model's forward pass is designed to return the loss directly
            loss = model(sentences_to_process, loss_fn)

            if loss is not None and loss.item() > 0:
                loss.backward()
                # Clip gradients to prevent exploding gradients, a common practice
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_epoch_loss += loss.item()

            if (i + 1) % 1 == 0:  # Print loss for every document since our dataset is tiny
                print(f'   Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Doc [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

        avg_epoch_loss = total_epoch_loss / len(data_loader) if len(data_loader) > 0 else 0
        print(f'--- End of Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Average Loss: {avg_epoch_loss:.4f} ---')

    print("4. Training finished.")

if __name__ == '__main__':
    train()
