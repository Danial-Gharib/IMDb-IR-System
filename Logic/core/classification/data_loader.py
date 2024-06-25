import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from word_embedding.fasttext_model import FastText
from word_embedding.fasttext_data_loader import preprocess_text

class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self, train=True, model_path=None, train_epochs=15, write_rs=True, save_model=False):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        if train:
            df = pd.read_csv(self.file_path)
            for review, sentiment in tqdm(zip(df['review'], df['sentiment']), desc='loading reviews and sentiments'):
                tokens = preprocess_text(review).split()
                self.review_tokens.append(tokens)
                self.sentiments.append(sentiment)
            label_encoder = LabelEncoder()
            self.sentiments = label_encoder.fit_transform(self.sentiments)
            if write_rs:
                self.write_reviews_sentiments()
            self.fasttext_model.prepare("classification_training/review_tokens.txt", "train", epoch=train_epochs)
            
            if save_model:
                self.fasttext_model.save_model(path=model_path)
        
        else:
            self.load_reviews_sentiments()
            self.fasttext_model.load_model(model_path)
        


    def write_reviews_sentiments(self, path='classification_training/'):

        with open(path + "review_tokens.txt", "w") as f:
            for tokens in self.review_tokens:
                for token in tokens:
                    f.write(token + " ")
                f.write("\n")
        with open(path + "sentiments.txt", "w") as f:
            for s in self.sentiments:
                f.write(str(s) + "\n")

    def load_reviews_sentiments(self, path='classification_training/'):

        with open(path + "review_tokens.txt", "r") as f:
            self.review_tokens = [line.strip().split(" ") for line in f]
        with open(path + "sentiments.txt", "r") as f:
            self.sentiments = list(map(int, f.read().strip().split("\n")))


    def get_embeddings(self, load=False, save=True):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        if load:
            self.load_embeddings()
            print("embeddings loaded successfully")
            return
        if len(self.review_tokens) == 0 or  len(self.sentiments) == 0:
            self.load_reviews_sentiments()
        for tokens in tqdm(self.review_tokens, desc='Generating embeddings'):
            emb = self.fasttext_model.get_query_embedding(" ".join(tokens))
            self.embeddings.append(emb)
        if save:
            self.save_embeddings()
            print("embeddings saved")
    
    def save_embeddings(self, path="classification_training/embeddings.npy"):

        np.save(path, self.embeddings)

    def load_embeddings(self, path="classification_training/embeddings.npy"):

        self.embeddings = np.load(path, allow_pickle=True).tolist()

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.embeddings, self.sentiments, test_size=test_data_ratio, random_state=42
        )
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    file_path = 'classification_dataset/archive/IMDB Dataset.csv'
    # df = pd.read_csv(file_path)
    # print(df.head())
    loader = ReviewLoader(file_path)
    loader.load_data(train=False, model_path="classification_training/ft_model.bin", write_rs=False, save_model=False)
    print("fasttext model loaded")
    loader.get_embeddings(save=True)
    print("embeddings generated")
    x_train, x_test, y_train, y_test = loader.split_data()
    print("Data has been loaded and split into training and testing sets.")
