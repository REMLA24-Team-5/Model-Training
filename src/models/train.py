"""Module that trains model given training data."""
import os
from src.models.model_definition import get_model
from joblib import dump, load

def train(file_folder):
    """
    Gets the model and trains it given the training and validation data.
    """

    # Get model
    model, params = get_model(os.path.join(file_folder, 'char_index.joblib'))

    # Load data
    x_train = load(os.path.join(file_folder,'x_train.joblib'))
    y_train = load(os.path.join(file_folder, 'y_train.joblib'))
    x_val = load(os.path.join(file_folder,'x_val.joblib'))
    y_val = load(os.path.join(file_folder,'y_val.joblib'))

    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )
    # Save model
    model.save(os.path.join(file_folder, 'model.h5'))
    model.save('test/data/model.h5')
    return model

if __name__ == "__main__":
    train('output')
