import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes, counts = np.unique(y, return_counts=True)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior = counts / self.number_of_samples

        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for idx, cls in enumerate(tqdm(self.classes, desc="Fitting the NaiveBayes model")):
            x_class = x[y == cls]
            self.feature_probabilities[idx, :] = (np.sum(x_class, axis=0) + self.alpha) / (x_class.shape[0] + self.alpha * self.number_of_features)

        self.log_probs = np.log(self.feature_probabilities)
        self.log_prior = np.log(self.prior)

        return self

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
        if isinstance(x, np.ndarray):
            log_likelihood = np.dot(x, self.log_probs.T)
        else:
            log_likelihood = x.dot(self.log_probs.T)
        log_likelihood += self.log_prior
        return self.classes[np.argmax(log_likelihood, axis=1)]

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        transformed_sentences = self.cv.transform(sentences)
        predictions = self.predict(transformed_sentences)
        percent_positive = np.mean(predictions == "positive") * 100
        return percent_positive


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    file_path = "classification_training/archive/IMDB Dataset.csv"
    loader = ReviewLoader(file_path=file_path)
    loader.load_data(train=False, model_path="classification_training/ft_model.bin", write_rs=False, save_model=False)
    print("model loaded")
    reviews, labels = loader.review_tokens, loader.sentiments
    print("reviews and sentiments loaded")

    reviews = [" ".join(review) for review in reviews]
    print("review tokens made back to sentences")

    x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.1, random_state=42)
    print("Data has been split")

    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)
    print("Count vectors generated")

    nb = NaiveBayes(count_vectorizer=cv)
    nb.fit(x_train_cv, y_train)
    print("Training Finished")

    report = nb.prediction_report(x_test_cv, y_test)
    print("Classification report\n", report)
    

    ##### results ########################
#     Classification report
#                precision    recall  f1-score   support

#            0       0.88      0.85      0.86      2481
#            1       0.85      0.89      0.87      2519

#     accuracy                           0.87      5000
#    macro avg       0.87      0.87      0.87      5000
# weighted avg       0.87      0.87      0.87      5000
