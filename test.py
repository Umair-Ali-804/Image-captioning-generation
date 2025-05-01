import torch
from model import EncoderCNN, DecoderRNN
from tokenizer import tokenizer, vocab  # Assume vocab has word2idx and idx2word
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def generate_caption(image_tensor, encoder, decoder, max_len=20):
    with torch.no_grad():
        feature = encoder(image_tensor)
        inputs = feature.unsqueeze(1)
        states = None
        caption = []

        for _ in range(max_len):
            hiddens, states = decoder.lstm(inputs, states)
            output = decoder.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            word = vocab.idx2word[predicted.item()]
            if word == "<end>":
                break
            caption.append(word)
            inputs = decoder.embed(predicted).unsqueeze(1)
        return " ".join(caption)

def main():
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, len(vocab)).to(device)

    encoder.load_state_dict(torch.load("weights/encoder.pth"))
    decoder.load_state_dict(torch.load("weights/decoder.pth"))

    image_tensor = load_image("sample.jpg")
    print(generate_caption(image_tensor, encoder, decoder))

if __name__ == "__main__":
    main()
