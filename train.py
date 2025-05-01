import torch
from torch import nn, optim
from model import EncoderCNN, DecoderRNN
from data_utils import get_loader
from tokenizer import tokenizer, vocab_size  # Assume a tokenizer.py exists
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    dataloader = get_loader("data/train.csv", "data/images", tokenizer, batch_size=32)
    encoder = EncoderCNN(embed_size=256).to(device)
    decoder = DecoderRNN(256, 512, vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    for epoch in range(10):
        for i, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}], Step [{i}], Loss: {loss.item():.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(encoder.state_dict(), "weights/encoder.pth")
    torch.save(decoder.state_dict(), "weights/decoder.pth")

if __name__ == "__main__":
    train_model()
