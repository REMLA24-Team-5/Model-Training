
import importlib
import shutil
import pytest
import gdown
import os
import random
import string
import tensorflow as tf
import numpy as np

from joblib import load
from keras.layers import Conv1D, MaxPooling1D, Dropout
from src.data.get_data import get_data
from src.features.process_data import process_data
from src.models.model_definition import get_model
from src.models.train import train
from src.models.predict import evaluate
from keras.models import load_model
pre_process = importlib.import_module('lib-ml.pre_process')


def fetch_preprocessing():
    """
    Fetches the pre-processing module from Google Drive
    """

    train = [line.strip() for line in open('test/data/train.txt', "r", encoding="utf-8").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open('test/data/test.txt', "r", encoding="utf-8").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]

    val=[line.strip() for line in open('test/data/val.txt', "r", encoding="utf-8").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]

    preprocessor = pre_process.Preprocessing(raw_x_train, raw_y_train, raw_x_test, raw_x_val)



    return preprocessor

def create_joblibs():
    if not os.path.exists('test/data'):
        os.mkdir('test/data')
    get_data('test/data', 'test/data')
    process_data('test/data')

def mutate_string(s, start_idx, end_idx):
    # Ensure start_idx and end_idx are within the bounds of the string
    start_idx = max(0, start_idx)
    end_idx = min(len(s), end_idx)
    
    if start_idx >= end_idx:
        raise ValueError("start_idx should be less than end_idx")
    
    # Convert the string to a list for easy mutation
    s_list = list(s)
    
    for i in range(start_idx, end_idx):
        s_list[i] = random.choice(string.printable)
    
    mutated_string = ''.join(s_list)
    
    return mutated_string

@pytest.fixture(scope="session", autouse=True)
def preprocessor():
    create_joblibs()
    preprocessor = fetch_preprocessing()
    yield preprocessor  # Provide the data to the tests
    shutil.rmtree('test/data')



def test_model_definition(preprocessor):
    model, params = get_model(os.path.join('test/data', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    conv1d_count = sum(1 for layer in model.layers if isinstance(layer, Conv1D))
    dropout_count = sum(1 for layer in model.layers if isinstance(layer, Dropout))
    pool_count = sum(1 for layer in model.layers if isinstance(layer, MaxPooling1D))
    expected_conv1d_count = 7  # Based on your model definition    conv1d_count = sum(1 for layer in model.layers if isinstance(layer, Conv1D))
    expected_dropout_count = 7
    expected_pool_count = 4

    assert conv1d_count == expected_conv1d_count
    assert dropout_count == expected_dropout_count
    assert pool_count == expected_pool_count

def test_train(preprocessor):
    model = train('test/model', 'test/data')
    input = "http://txcvg.h.zz.zxx"
    input_preprocessed = preprocessor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_preprocessed, batch_size=1)

    # Convert predicted probabilities to binary labels
    prediction_binary = (np.array(prediction) > 0.5).astype(int)

    assert prediction_binary == 1

def test_evaluate(preprocessor):
    acc = evaluate('test/model', 'test/data')
    assert acc > 0

def test_predict_legitimate(preprocessor):
    model, params = get_model(os.path.join('test/data', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    input = "https://google.com"
    input_preprocessed = preprocessor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_preprocessed, batch_size=1)

    # Convert predicted probabilities to binary labels
    prediction_binary = (np.array(prediction) > 0.5).astype(int)

    assert prediction_binary == 0

def test_predict_phising(preprocessor):
    model, params = get_model(os.path.join('test/model', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    input = "http://txcvg.h.zz.zxx"
    input_preprocessed = preprocessor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_preprocessed, batch_size=1)

    # Convert predicted probabilities to binary labels
    prediction_binary = (np.array(prediction) > 0.5).astype(int)

    assert prediction_binary == 1


def test_mutamorphic_legit(preprocessor):
    model, params = get_model(os.path.join('test/model', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    input = "http://google.com"

    for i in range(10):
        test_string = mutate_string(input, 9, 11)
        test_preprocessed = preprocessor.process_URL(test_string).reshape(1,200,1)

        prediction = model.predict(test_preprocessed, batch_size=1)
        prediction_binary = (np.array(prediction) > 0.5).astype(int)
        assert prediction_binary == 0


def test_mutamorphic_phising(preprocessor):
    model, params = get_model(os.path.join('test/model', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    input = "http://txcvg.h.zz.zxx.hhhh"

    for i in range(10):
        test_string = mutate_string(input,0, 10)
        test_preprocessed = preprocessor.process_URL(test_string).reshape(1,200,1)

        prediction = model.predict(test_preprocessed, batch_size=1)
        prediction_binary = (np.array(prediction) > 0.5).astype(int)
        assert prediction_binary == 1



def test_non_determinisim(preprocessor):
    np.random.seed(42)
    tf.random.set_seed(42)

    model, params = get_model(os.path.join('test/model', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    input = "https://google.com"
    input_preprocessed = preprocessor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction_1 = model.predict(input_preprocessed, batch_size=1)

    np.random.seed(11)
    tf.random.set_seed(11)

    model, params = get_model(os.path.join('test/model', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    input = "https://google.com"
    input_preprocessed = preprocessor.process_URL(input).reshape(1,200,1)

    # Make predictions using the pre-trained model
    prediction_2 = model.predict(input_preprocessed, batch_size=1)
    assert np.allclose(prediction_1, prediction_2, atol=0.1), "Big accuracy difference"


def test_pipeline(preprocessor):
    get_data('test/data', 'test/data')
    process_data('test/data')
    preprocessor = fetch_preprocessing()
    model, params = get_model(os.path.join('test/data', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    inputs = ["https://google.com",
              "http://www.youtube.com",
              "http://tudelft.nl"]
    
    gt = [0,0,0]
    acc = 0

    for index, input in enumerate(inputs):
        input_preprocessed = preprocessor.process_URL(input).reshape(1,200,1) 
        pred = model.predict(input_preprocessed, batch_size=1)
        pred = (np.array(pred) > 0.5).astype(int)
        if pred == gt[index]:
            acc += 1
    acc = acc / 3
    assert acc > 0
