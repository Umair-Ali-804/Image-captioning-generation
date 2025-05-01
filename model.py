# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, GRU, Dropout, Concatenate

def build_model(vocab_size, max_length, embedding_matrix, embedding_dim):
    # Image feature extractor model
    image_input = Input(shape=(1536,))  # assuming EfficientNetB3 features
    image_dense = Dense(256, activation='relu')(image_input)
    image_dropout = Dropout(0.5)(image_dense)

    # Sequence model
    caption_input = Input(shape=(max_length,))
    caption_embed = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    )(caption_input)
    caption_gru = GRU(256)(caption_embed)
    caption_dropout = Dropout(0.5)(caption_gru)

    # Decoder model
    decoder = Concatenate()([image_dropout, caption_dropout])
    decoder_dense = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder_dense)

    model = Model(inputs=[image_input, caption_input], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model