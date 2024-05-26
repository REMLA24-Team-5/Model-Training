
import importlib
import pytest
import gdown
import numpy as np
import os
from joblib import load
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