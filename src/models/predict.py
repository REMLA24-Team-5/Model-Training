"""Module that classifys a given input"""
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import load
import numpy as np

def main():
    """
    Takes in input and predicts using model and outputs results.
    """
    np.random.seed()
    # Load model
    model = load('../../output/model.joblib')
    # Load test data
    x_test = load('../../output/x_test.joblib')
    y_test = load('../../output/y_test.joblib')

    y_pred = model.predict(x_test, batch_size=1000)
    print(y_pred)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test=y_test.reshape(-1,1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test,y_pred_binary))


    sns.heatmap(confusion_mat,annot=True)

if __name__ == "__main__":
    main()
