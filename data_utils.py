import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        caption = row['caption']

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        tokens = self.tokenizer(caption)

        return image, torch.tensor(tokens)

def get_loader(csv_path, img_dir, tokenizer, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ImageCaptionDataset(csv_path, img_dir, tokenizer, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
