"""Module that classifys a given input"""
import json
import os
from pathlib import Path
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import load
import numpy as np


path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


from src.models.model_definition import get_model


def evaluate(model_folder, joblib_folder, output_folder=None):
    """
    Takes in input and predicts using model and outputs results.
    """
    np.random.seed()
    # Load model
    model, params = get_model(os.path.join(model_folder, 'char_index.joblib'))
    model.load_weights(os.path.join(model_folder, 'model.h5'))
    # Load test data
    x_test = load(os.path.join(joblib_folder, 'x_test.joblib'))
    y_test = load(os.path.join(joblib_folder,'y_test.joblib'))

    # Collect all information into a dictionary
    output_dict = {}

    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test=y_test.reshape(-1,1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    output_dict['classification_report'] = report

    # Calculate confusion matrix and format as string
    confusion_mat = confusion_matrix(y_test, y_pred_binary)

    # Get the number of classes
    num_classes = confusion_mat.shape[0]

    # Initialize an empty string to store the confusion matrix
    confusion_mat_string = ""

    # Add header row
    confusion_mat_string += "Predicted" + " " * 8 + "|"
    for i in range(num_classes):
        confusion_mat_string += f" Class {i}" + " " * 4 + "|"
    confusion_mat_string += "\n" + "-" * 72 + "\n"

    # Add rows for each class
    for i in range(num_classes):
        confusion_mat_string += f" Actual Class {i}" + " " * (12 - len(f" Actual Class {i}")) + "|"
        for j in range(num_classes):
            entry = confusion_mat[i, j]
            confusion_mat_string += f" {entry}" + " " * (10 - len(str(entry))) + "|"
        confusion_mat_string += "\n"

    # Store confusion matrix string in the output dictionary
    output_dict['confusion_matrix'] = confusion_mat_string

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_binary)
    output_dict['accuracy'] = accuracy

    # Dump collected information to a JSON file
    if output_folder:
        with open(os.path.join(output_folder, 'metrics.json'), 'w', encoding="utf-8") as json_file:
            json.dump(output_dict, json_file, indent=4)

    return accuracy

if __name__ == "__main__":
    evaluate("output", "output")
