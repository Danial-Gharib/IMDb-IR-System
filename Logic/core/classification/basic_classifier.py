import numpy as np
from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        self.ft = FastText()

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        
        embeddings = [self.ft.get_query_embedding(sentence) for sentence in tqdm(sentences, desc='get positive reviews percentage', mininterval=0.1, ncols=None)]
        predictions = self.predict(np.array(embeddings))

        num_positive_reviews = np.sum(predictions == 1)
        total_reviews = len(sentences)
        if total_reviews == 0:
            return 0.0
        percent_positive = (num_positive_reviews / total_reviews) * 100
        return percent_positive

