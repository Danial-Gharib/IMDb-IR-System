import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import json
# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if text is None:
        return ""
    if lower_case:
        text = text.lower()
    if punctuation_removal:
        text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    if stopword_removal:
        stop_words = set(stopwords.words('english') + stopwords_domain)
        tokens = [word for word in tokens if word not in stop_words]
    
    tokens = [word for word in tokens if len(word) >= minimum_length]

    return ' '.join(tokens)


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        data = None
        with open(self.file_path) as f:
            data = json.load(f)
        
        titles = []
        genres = []
        summaries = []
        reviews = []
        synopsis = []
        
        for i, movie in enumerate (tqdm(data, desc='Creating Dataframe', mininterval=0.1, ncols=None)):
            titles.append(preprocess_text(movie.get("title", "")))
            genres.append(preprocess_text(" ".join(movie.get('genres', []) or [])))
            summaries.append(preprocess_text(" ".join(movie.get('summaries', []) or [])))
            synopsis.append(preprocess_text(" ".join(movie.get('synopsis', []) or [])))
            # print(movie.get('reviews'))
            movie_reviews = movie.get('reviews', [])
            movie_review = []
            if movie_reviews is not None:
                for review in movie_reviews:
                    movie_review.append(review[0])
                    movie_review.append(review[1])
            reviews.append(preprocess_text(" ".join(movie_review)))

            if (i + 1) % 100 == 0:
                print(f"{i+1} movies added up to now.")

        df = pd.DataFrame({
            'synopsis': synopsis,
            'summaries': summaries,
            'reviews': reviews,
            'title': titles,
            'genre': genres
        })
        return df

    def create_train_data(self, save=True):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = pd.read_csv("fasttext_training/ft.csv")
        df['synopsis'] = df['synopsis'].fillna('').astype(str)
        df['summaries'] = df['summaries'].fillna('').astype(str)
        df['reviews'] = df['reviews'].fillna('').astype(str)
        df['title'] = df['title'].fillna('').astype(str)
        df['training'] = df['synopsis'] + ' ' + df['summaries'] + ' ' + df['reviews'] + ' ' + df['title']
        y = df['genre'].fillna('').astype(str)
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
        ### now we write it as .txt:
        with open("fasttext_training/ft.txt", "w") as f:
            for text in df['training']:
                f.write(text + "\n")
        X = df['training'].to_numpy()
        if save:
            self.save_traindata(df)
        return X, y
    
    def save_traindata(self, df, path="fasttext_training/clustering_training.csv"):
        df.to_csv(path)

    def load_traindata(self, path="fasttext_training/clustering_training.csv"):
        df = pd.read_csv(path)
        df['genre'] = df['genre'].fillna('').apply(lambda x: x.split())
        df['first_genre'] = df['genre'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
        
        # Initialize the LabelEncoder
        labelencoder = LabelEncoder()
        
        # Fit and transform the first genres
        y = labelencoder.fit_transform(df['first_genre'])
        
        # Print the number of unique labels
        print(f"Number of unique first genres: {len(set(y))}")

        # unique_genre_combinations = np.unique(y, axis=0)
        # print(f"Number of unique genre combinations: {len(unique_genre_combinations)}")

        return df['training'].to_numpy(), y

if __name__ == '__main__':
    # nltk.download('stopwords')
    print("HI")
    path = 'IMDB_crawled_standard.json'
    dataloader = FastTextDataLoader(path)
    df = dataloader.read_data_to_df()
    df.to_csv("fasttext_training/ft.csv")
    print("training data made and save.")