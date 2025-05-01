from tensorflow.keras.models import load_model
from data_utils import preprocess_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_caption(model, tokenizer, feature_model, image_path, max_length):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    feature = feature_model.predict(img)
    caption = ['<start>']
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([' '.join(caption)])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([feature, seq], verbose=0)
        next_word = tokenizer.index_word[np.argmax(yhat)]
        if next_word == '<end>':
            break
        caption.append(next_word)
    return ' '.join(caption[1:])

def evaluate(image_path):
    model = load_model('weights/best_model.h5')
    # Assume tokenizer and feature_model are loaded in global context or reload here
    caption = generate_caption(model, tokenizer, feature_model, image_path, max_length)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()