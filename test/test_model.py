
import importlib
import shutil
import pytest
import gdown
import numpy as np
import os
from joblib import load
from keras.layers import Conv1D, MaxPooling1D, Dropout
from src.data.get_data import get_data
from src.models.model_definition import get_model

pre_process = importlib.import_module('lib-ml.pre_process')


def fetch_preprocessing():
    """
    Fetches the pre-processing module from Google Drive
    """

    gdown.download_folder('https://drive.google.com/drive/folders/1_NobSEMZS8jogSAEZ9ZLBUTemPiASJRg',
                          output="test/data", quiet=False)
    
    train = [line.strip() for line in open('test/data/train.txt', "r", encoding="utf-8").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open('test/data/test.txt', "r", encoding="utf-8").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]

    val=[line.strip() for line in open('test/data/val.txt', "r", encoding="utf-8").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]

    preprocessor = pre_process.Preprocessing(raw_x_train, raw_y_train, raw_x_test, raw_x_val)

    return preprocessor


@pytest.fixture(scope="session", autouse=True)
def model():
    model = load('test/data/model.joblib')
    preprocessor = fetch_preprocessing()
    yield model, preprocessor  # Provide the data to the tests
    os.remove('test/data/train.txt')
    os.remove('test/data/test.txt')
    os.remove('test/data/val.txt')


def test_model_definition(model):
    model, params = get_model(os.path.join('test/data', 'char_index.joblib'))
    conv1d_count = sum(1 for layer in model.layers if isinstance(layer, Conv1D))
    dropout_count = sum(1 for layer in model.layers if isinstance(layer, Dropout))
    pool_count = sum(1 for layer in model.layers if isinstance(layer, MaxPooling1D))
    expected_conv1d_count = 7  # Based on your model definition    conv1d_count = sum(1 for layer in model.layers if isinstance(layer, Conv1D))
    expected_dropout_count = 7
    expected_pool_count = 4

    assert conv1d_count == expected_conv1d_count
    assert dropout_count == expected_dropout_count
    assert pool_count == expected_pool_count


def test_predict_legitimate(model):
    model, preprocesor = model
    input = "https://google.com"
    input_preprocessed = preprocesor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_preprocessed, batch_size=1)

    # Convert predicted probabilities to binary labels
    prediction_binary = (np.array(prediction) > 0.5).astype(int)

    assert prediction_binary == 0

def test_predict_phising(model):
    model, preprocesor = model
    input = "http://txcvg.h.zz.zxx"
    input_preprocessed = preprocesor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_preprocessed, batch_size=1)

    # Convert predicted probabilities to binary labels
    prediction_binary = (np.array(prediction) > 0.5).astype(int)

    assert prediction_binary == 1


def test_mutamorphic(model):
    model, preprocesor = model

    input_1 = "http://google.com"
    input_1_preprocessed = preprocesor.process_URL(input_1).reshape(1,200,1)

    input_2 = "https://google.nl"
    input_2_preprocessed = preprocesor.process_URL(input_2).reshape(1,200,1)
    # Make predictions using the pre-trained model
    prediction_1 = model.predict(input_1_preprocessed, batch_size=1)
    prediction_binary_1 = (np.array(prediction_1) > 0.5).astype(int)

    prediction_2 = model.predict(input_2_preprocessed, batch_size=1)
    prediction_binary_2 = (np.array(prediction_2) > 0.5).astype(int)

    assert prediction_binary_1 == prediction_binary_2
