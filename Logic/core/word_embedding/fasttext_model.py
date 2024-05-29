import fasttext
import re
from tqdm import tqdm
from scipy.spatial import distance

from fasttext_data_loader import FastTextDataLoader



class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None


    def train(self, texts, epoch=5):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        #TODO
        self.model = fasttext.train_unsupervised(texts, model=self.method, epoch=epoch)

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        # TODO
        vec1 = self.model.get_word_vector(word1)
        vec2 = self.model.get_word_vector(word2)
        vec3 = self.model.get_word_vector(word3)
        # Perform vector arithmetic
        # TODO
        result_vec = vec2 - vec1 + vec3
        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # TODO
        word_vectors = {word: self.model.get_word_vector(word) for word in self.model.words}
        # Exclude the input words from the possible results
        # TODO
        best_word = None
        min_dist = float('inf')
        # Find the word whose vector is closest to the result vector
        # TODO
        for word, vector in word_vectors.items():
            if word in {word1, word2, word3}:
                continue
            dist = distance.cosine(result_vec, vector)
            if dist < min_dist:
                min_dist = dist
                best_word = word
        return best_word

    def save_model(self, path='fasttext_training/FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path=path)

    def load_model(self, path="fasttext_training/FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path=path)

    def prepare(self, dataset, mode, save=False, path='fasttext_training/FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":

    ft_model = FastText(method='skipgram')

    path = 'IMDB_crawled_standard.json'
    ft_data_loader = FastTextDataLoader(path)

    X, y = ft_data_loader.create_train_data()
    print("training data created.")
    ft_model.train("fasttext_training/ft.txt", epoch=40)
    print("model trained")
    ft_model.prepare(None, mode = "save", save=True)
    print("model saved")
    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.model.get_analogies(word1, word2, word3)[0][1]}")
