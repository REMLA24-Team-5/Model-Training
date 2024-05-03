from model_definition import getModel
from model_params import params
from joblib import dump, load

def main():

    # Get model
    model = getModel()

    # Load data
    x_train = load('../../output/x_train.joblib')
    y_train = load('../../output/y_train.joblib')
    x_val = load('../../output/x_val.joblib')
    y_val = load('../../output/y_val.joblib')

    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )
    # Save model
    dump(model, '../../output/model.joblib')

if __name__ == "__main__":
    main()