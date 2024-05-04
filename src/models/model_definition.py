"""Module that defines the model used for training"""
"""Module that defines the model used for training"""
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from joblib import load

def get_model():
    """
    Builds the model from scratch using the keras library.

    Returns:
        Sequential: Defined sequential model
    """
def get_model():
    """
    Builds the model from scratch using the keras library.

    Returns:
        Sequential: Defined sequential model
    """
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': 5000,
              'batch_test': 5000,
              'categories': ['phishing', 'legitimate'],
              'char_index': None,
              'epoch': 30,
              'embedding_dimension': 50,
              'dataset_dir': "../dataset/small_dataset/"}

    model = Sequential()
    char_index = load('output/char_index.joblib')
    voc_size = len(char_index.keys())
    print(f"voc_size: {voc_size}")
    model.add(Embedding(voc_size + 1, 50, input_length=200)) # Look into this

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))
    return model, params