import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC(verbose=True)

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        return self.model.predict(x)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        preds = self.predict(x)
        return classification_report(y, preds)


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = "classification_dataset/archive/IMDB Dataset.csv"
    loader = ReviewLoader(file_path=file_path)
    loader.load_data(train=False, model_path="classification_training/ft_model.bin", write_rs=False, save_model=False)
    print("model loaded")
    loader.get_embeddings(load=True, save=False)
    print("embeddings generated")
    x_train, x_test, y_train, y_test = loader.split_data()
    # print(type(y_train[0]))
    # print(type(x_train[0]))
    print("Data has been split")
    svm = SVMClassifier()
    svm.fit(x_train, y_train)
    print("Training finished")
    report = svm.prediction_report(x_test, y_test)
    print("Classification report\n", report)

    ############## report ####################
#     Classification report
#                precision    recall  f1-score   support

#            0       0.89      0.88      0.89      4961
#            1       0.88      0.89      0.89      5039

#     accuracy                           0.89     10000
#    macro avg       0.89      0.89      0.89     10000
# weighted avg       0.89      0.89      0.89     10000