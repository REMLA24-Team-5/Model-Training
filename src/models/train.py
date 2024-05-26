"""Module that trains model given training data."""
from model_definition import get_model
from joblib import dump, load

def main():
    """
    Gets the model and trains it given the training and validation data.
    """

    # Get model
    model, params = get_model()

    # Load data
    x_train = load('output/x_train.joblib')
    y_train = load('output/y_train.joblib')
    x_val = load('output/x_val.joblib')
    y_val = load('output/y_val.joblib')

    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )
    # Save model
    dump(model, 'output/model.joblib')
    dump(model, 'test/data/model.joblib')

if __name__ == "__main__":
    main()
