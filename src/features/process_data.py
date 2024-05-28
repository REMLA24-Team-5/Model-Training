"""Module processing data and repersisting it."""
from joblib import dump, load
import importlib
pre_process = importlib.import_module('lib-ml.pre_process')

# pylint: disable=too-many-locals



def process_data():
    """
    Loads data from joblibs, processes it and stores it again into joblib files.
    """
    # Load data
    raw_x_train = load('output/raw_x_train.joblib')
    raw_x_val = load('output/raw_x_val.joblib')
    raw_x_test = load('output/raw_x_test.joblib')
    raw_y_train = load('output/raw_y_train.joblib')
    raw_y_val = load('output/raw_y_val.joblib')
    raw_y_test = load('output/raw_y_test.joblib')

    preprocessor = pre_process.Preprocessing(raw_x_train, raw_y_train, raw_x_test, raw_x_val)
    x_train = preprocessor.process_dataset(raw_x_train)
    x_val = preprocessor.process_dataset(raw_x_val)
    x_test = preprocessor.process_dataset(raw_x_test)

    # Store processed data
    dump(x_train, 'output/x_train.joblib')
    dump(x_val, 'output/x_val.joblib')
    dump(x_test, 'output/x_test.joblib')
    dump(preprocessor.get_char_index(), 'output/char_index.joblib')

    y_train = preprocessor.process_labels(raw_y_train)
    y_val = preprocessor.process_labels(raw_y_val)
    y_test = preprocessor.process_labels(raw_y_test)

    # Store processed data
    dump(y_train, 'output/y_train.joblib')
    dump(y_val, 'output/y_val.joblib')
    dump(y_test, 'output/y_test.joblib')

if __name__ == "__main__":
    process_data()
