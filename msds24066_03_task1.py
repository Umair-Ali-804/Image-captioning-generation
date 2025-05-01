# Combines data loading, training, testing, and caption generation
from train import train_model
from test import main as test_model

if __name__ == "__main__":
    print("Training model...")
    train_model()
    print("Testing model...")
    test_model()
