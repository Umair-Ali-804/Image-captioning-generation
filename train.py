import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(model, X_img_train, X_seq_train, y_train, X_img_val, X_seq_val, y_val, output_dir='weights'):
    callbacks = [
        ModelCheckpoint(os.path.join(output_dir, 'best_model.h5'), save_best_only=True),
        ReduceLROnPlateau(patience=3),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        [X_img_train, X_seq_train], y_train,
        validation_data=([X_img_val, X_seq_val], y_val),
        batch_size=32,
        epochs=30,
        callbacks=callbacks
    )

    # Save training plot
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('graphs/loss_plot.png')