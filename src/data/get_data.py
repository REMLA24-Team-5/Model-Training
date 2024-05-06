"""Module retrieving the data and persisting it to single files."""
import sys
from joblib import dump

# pylint: disable=consider-using-with

def main():
    """
    Polls data and persists into single joblib files.
    """
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/data/get_data.py data/train.txt "
                         "data/test.txt data/val.txt\n")
        sys.exit(1)

    train_data = sys.argv[1]
    test_data = sys.argv[2]
    val_data = sys.argv[3]

    train = [line.strip() for line in open(train_data, "r", encoding="utf-8").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]
    dump(raw_x_train, 'output/raw_x_train.joblib')
    dump(raw_y_train, 'output/raw_y_train.joblib')


    test = [line.strip() for line in open(test_data, "r", encoding="utf-8").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]
    dump(raw_x_test, 'output/raw_x_test.joblib')
    dump(raw_y_test, 'output/raw_y_test.joblib')

    val=[line.strip() for line in open(val_data, "r", encoding="utf-8").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]
    dump(raw_x_val, 'output/raw_x_val.joblib')
    dump(raw_y_val, 'output/raw_y_val.joblib')

if __name__ == "__main__":
    main()
