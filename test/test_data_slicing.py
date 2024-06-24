import os
import shutil
import random
import importlib
from src.data.get_data import get_data
from joblib import dump,load
from src.features.process_data import process_data
from src.models.model_definition import get_model
from sklearn.metrics import accuracy_score
import numpy as np
pre_process = importlib.import_module('lib-ml.pre_process')


acc_threshold = 0.5

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


def test_data_slicing_1():
    if not os.path.exists('test/output'):
        os.mkdir('test/output')

    get_data('test/data', 'test/output')
    preprocessor = process_data('test/output')
    model, params = get_model(os.path.join('test/model', 'char_index.joblib'))
    model.load_weights('test/model/model.h5')
    x_test = load('test/output/x_test.joblib')
    y_test = load('test/output/y_test.joblib')
    
    entries = max(len(x_test), 500)
    end = random.randint(0, entries) # nosec B311
    start = random.randint(0, end - 1)# nosec B311
    
    slice_x = x_test[start:end]
    slice_y = y_test[start:end]
    
    acc_slice = 0
    prediction = model.predict(slice_x, batch_size=1000)
    pred_binary = (np.array(prediction) > 0.5).astype(int)
    y_test=slice_y.reshape(-1,1)
    accuracy = accuracy_score(y_test, pred_binary)
    assert accuracy > acc_threshold, "New model is not robust to data slicing because of low accuracy"
    shutil.rmtree('test/output')
    
    
    