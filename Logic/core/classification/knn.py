import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from scipy.stats import mode
from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.train_data = None
        self.train_labels = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.train_data = np.array(x)
        self.train_labels = np.array(y)

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
        x = np.array(x)
        predictions = []
        for i in tqdm(range(x.shape[0]), desc="Predictiong"):
            distances = np.sqrt(np.sum((self.train_data - x[i]) ** 2, axis=1))
            k_nearest_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.train_labels[k_nearest_indices]
            predicted_label = np.argmax(np.bincount(k_nearest_labels))
            # print(predicted_label)
            predictions.append(predicted_label)
        return np.array(predictions)

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
        report = classification_report(y, preds)
        return report


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = "classification_training/archive/IMDB Dataset.csv"
    loader = ReviewLoader(file_path=file_path)
    loader.load_data(train=False, model_path="classification_training/ft_model.bin", write_rs=False, save_model=False)
    print("fasttext model loaded")
    loader.get_embeddings(load=True, save=False)
    print("embeddings generated")
    x_train, x_test, y_train, y_test = loader.split_data()
    print("Data has been split")

    ###
    classifier = KnnClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    print("Training fininshed")
    report = classifier.prediction_report(x_test, y_test)
    print("Classification report\n", report)

    ##############  results ####################
#     Classification report
#                precision    recall  f1-score   support

#            0       0.82      0.86      0.84      4961
#            1       0.86      0.82      0.84      5039

#     accuracy                           0.84     10000
#    macro avg       0.84      0.84      0.84     10000
# weighted avg       0.84      0.84      0.84     10000

