from joblib import dump

def main():
    train = [line.strip() for line in open("data/train.txt", "r").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]
    dump(raw_x_train, 'output/raw_x_train.joblib')
    dump(raw_y_train, 'output/raw_y_train.joblib')

    test = [line.strip() for line in open("data/test.txt", "r").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]
    dump(raw_x_test, 'output/raw_x_test.joblib')
    dump(raw_y_test, 'output/raw_y_test.joblib')

    val=[line.strip() for line in open("data/val.txt", "r").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]
    dump(raw_x_val, 'output/raw_x_val.joblib')
    dump(raw_y_val, 'output/raw_y_val.joblib')

if __name__ == "__main__":
    main()

