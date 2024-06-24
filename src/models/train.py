"""Module that trains model given training data."""
import os
from src.models.model_definition import get_model
from joblib import dump, load

def train(char_index_folder, joblib_folder, output_folder=None,output_test=False):
    """
    Gets the model and trains it given the training and validation data.
    """

    # Get model
    model, params = get_model(os.path.join(char_index_folder, 'char_index.joblib'))

    # Load data
    x_train = load(os.path.join(joblib_folder,'x_train.joblib'))
    y_train = load(os.path.join(joblib_folder, 'y_train.joblib'))
    x_val = load(os.path.join(joblib_folder,'x_val.joblib'))
    y_val = load(os.path.join(joblib_folder,'y_val.joblib'))

    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )
    # Save model
    if output_folder:
        model.save(os.path.join(output_folder, 'model.h5'))

    if output_test:
        model.save('test/model/model.h5')
    return model

if __name__ == "__main__":
    train('output', 'output', 'output')
