from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import dump, load

def main():
    # Load data
    raw_x_train = load('output/raw_x_train.joblib')
    raw_x_val = load('output/raw_x_val.joblib')
    raw_x_test = load('output/raw_x_test.joblib')
    raw_y_train = load('output/raw_y_train.joblib')
    raw_y_val = load('output/raw_y_val.joblib')
    raw_y_test = load('output/raw_y_test.joblib')

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    dump(char_index, 'output/char_index.joblib')
    sequence_length=200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    # Store processed data
    dump(x_train, 'output/x_train.joblib')
    dump(x_val, 'output/x_val.joblib')
    dump(x_test, 'output/x_test.joblib')

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    # Store processed data
    dump(y_train, 'output/y_train.joblib')
    dump(y_val, 'output/y_val.joblib')
    dump(y_test, 'output/y_test.joblib')

if __name__ == "__main__":
    main()