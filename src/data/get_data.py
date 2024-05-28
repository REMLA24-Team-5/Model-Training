"""Module retrieving the data and persisting it to single files."""
import os
import sys
from joblib import dump
import gdown

# pylint: disable=consider-using-with

def download_from_drive(data_output_folder):
    gdown.download_folder('https://drive.google.com/drive/folders/1_NobSEMZS8jogSAEZ9ZLBUTemPiASJRg',
                          output=data_output_folder, quiet=False)

def get_data(training_file_dir, joblib_output_dir):
    """
    Polls data and persists into single joblib files.
    """

    train_data = os.path.join(training_file_dir, 'train.txt')
    test_data = os.path.join(training_file_dir, 'test.txt')
    val_data = os.path.join(training_file_dir, 'val.txt')

    download_from_drive(training_file_dir)


    train = [line.strip() for line in open(train_data, "r", encoding="utf-8").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]
    dump(raw_x_train, os.path.join(joblib_output_dir, 'raw_x_train.joblib'))
    dump(raw_y_train, os.path.join(joblib_output_dir, 'raw_y_train.joblib'))


    test = [line.strip() for line in open(test_data, "r", encoding="utf-8").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]
    dump(raw_x_test, os.path.join(joblib_output_dir, 'raw_x_test.joblib'))
    dump(raw_y_test, os.path.join(joblib_output_dir, 'raw_y_test.joblib'))

    val=[line.strip() for line in open(val_data, "r", encoding="utf-8").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]
    dump(raw_x_val, os.path.join(joblib_output_dir, 'raw_x_val.joblib'))
    dump(raw_y_val, os.path.join(joblib_output_dir, 'raw_y_val.joblib'))

if __name__ == "__main__":
    get_data('data', 'output')
