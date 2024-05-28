import shutil
import os

from src.data.get_data import get_data

def test_get_data():
    if os.path.exists('output'):
        shutil.rmtree('output')
    
    os.mkdir('output')
    get_data('data', 'output')
    assert os.path.exists(os.path.join('output', 'raw_x_train.joblib'))
    assert os.path.exists(os.path.join('output', 'raw_y_train.joblib'))

    assert os.path.exists(os.path.join('output', 'raw_x_test.joblib'))
    assert os.path.exists(os.path.join('output', 'raw_y_test.joblib'))

    assert os.path.exists(os.path.join('output', 'raw_x_val.joblib'))
    assert os.path.exists(os.path.join('output', 'raw_y_val.joblib'))
    shutil.rmtree('output')
    

