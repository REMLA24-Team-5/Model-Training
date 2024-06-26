import os
import shutil

import pytest
from src.data.get_data import get_data
from src.features.process_data import process_data


@pytest.fixture(scope="session", autouse=True)
def data_gathering():
    if os.path.exists('output'):
        shutil.rmtree('output')
    
    os.mkdir('output')
    get_data('data', 'output')
    yield 'output'
    shutil.rmtree('output')

def test_feature_processing_1(data_gathering):
    process_data('test/data')
    assert os.path.exists(os.path.join('test/data', 'x_train.joblib'))
    assert os.path.exists(os.path.join('test/data', 'x_val.joblib'))
    assert os.path.exists(os.path.join('test/data', 'x_test.joblib'))



    
