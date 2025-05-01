import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from gensim.models import KeyedVectors
from collections import Counter

class TokenizerHelper:
    def __init__(self, word2vec_path=None):
        # Load the Word2Vec model if provided
        if word2vec_path:
            self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            self.word2vec = None

    def build_tokenizer(self, captions, max_vocab_size=10000):
        tokenizer = get_tokenizer('basic_english')  # Using a basic English tokenizer from torchtext
        counter = Counter()

        # Tokenize each caption and count the frequency of each word
        for caption in captions:
            tokens = tokenizer(caption)
            counter.update(tokens)

        # Get the most common words up to the max_vocab_size limit
        vocab = [word for word, _ in counter.most_common(max_vocab_size)]

        # Add <unk> token if not already in vocab
        if '<unk>' not in vocab:
            vocab.append('<unk>')

        # Create a word index mapping
        word_index = {word: idx for idx, word in enumerate(vocab)}

        return word_index

    def create_embedding_matrix(self, word_index, embedding_dim=300):
        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        # For each word in the word index, check if it's in the Word2Vec model and get the embedding
        for word, idx in word_index.items():
            if word in self.word2vec:
                embedding_matrix[idx] = self.word2vec[word]
            else:
                # If the word isn't in Word2Vec, keep it as a zero vector (this could be customized further)
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

        # Convert to a PyTorch tensor
        return torch.tensor(embedding_matrix, dtype=torch.float32)
