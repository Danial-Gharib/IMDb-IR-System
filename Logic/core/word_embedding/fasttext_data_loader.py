import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utility.preprocess import Preprocessor


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
        df = pd.DataFrame({
            'synopsis': data['synopses'],
            'summaries': data['summaries'],
            'reviews': data['reviews'],
            'title': data['titles'],
            'genre': data['genres']
        })
        return df
        pass

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        X = df['synopsis'] + ' ' + df['summaries'] + ' ' + df['reviews'] + ' ' + df['title']
        y = df['genre']
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
        return X.to_numpy(), y
        pass


if __name__ == '__main__':
    print("HI")
    # path = 'indexes_standard/'
    # dataloader = FastTextDataLoader(path)
    # df = dataloader.read_data_to_df()
    # df.to_csv("fasttext/fasttext_training.csv")