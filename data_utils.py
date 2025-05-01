# data_utils.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gensim.models import Word2Vec
import re

def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return f"<start> {text.strip()} <end>"

def load_captions(caption_file):
    df = pd.read_csv(caption_file)
    df = df.dropna()
    df['caption'] = df['caption'].apply(clean_caption)
    return df

def build_tokenizer(captions):
    tokenizer = Tokenizer(oov_token="<unk>", filters='')
    tokenizer.fit_on_texts(captions)
    return tokenizer

def create_embedding_matrix(tokenizer, embedding_dim=300):
    w2v_model = Word2Vec([c.split() for c in tokenizer.texts], vector_size=embedding_dim, window=5, min_count=1)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
        else:
            embedding_matrix[idx] = np.random.uniform(-0.1, 0.1, embedding_dim)
    return embedding_matrix

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    return img

def extract_features(image_paths, model):
    features = {}
    for img_path in image_paths:
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img)
        features[os.path.basename(img_path)] = feature[0]
    return features

def create_sequences(tokenizer, max_length, descriptions, image_features, vocab_size):
    X_img, X_seq, y = [], [], []
    for key, desc in descriptions.items():
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            X_img.append(image_features[key])
            X_seq.append(in_seq)
            y.append(tf.keras.utils.to_categorical(out_seq, num_classes=vocab_size))
    return np.array(X_img), np.array(X_seq), np.array(y)
