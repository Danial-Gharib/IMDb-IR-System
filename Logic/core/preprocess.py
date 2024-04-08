import re
import nltk
import os
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class Preprocessor:

    def __init__(self, documents: list, stopwords_path: str, does_stem=True, does_lemmatize=True):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents
        stodwords_file = os.path.join(os.path.dirname(__file__), stopwords_path)
        with open(stodwords_file , 'r') as f:
            self.stopwords = set(f.read().splitlines())
        # nltk.download('punkt')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.doesStem = does_stem
        self.doesLemm = does_lemmatize

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
         # TODO
        preprocessed_docs = []
        for doc in self.documents:
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            doc = self.normalize(doc)
            doc = self.remove_stopwords(doc)
            preprocessed_docs.append(doc)
        return preprocessed_docs

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        # TODO
        text = text.lower()
        if self.doesStem:
            text = self.stem(text)
        if self.doesLemm:
            text = self.lemmatize(text)
        return text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        # TODO
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        # TODO
        return text.translate(str.maketrans('', '', string.punctuation)) 
    
    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        # TODO
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        # TODO
        words = word_tokenize(text)
        return [word for word in words if word not in self.stopwords]
    
    def lemmatize(self, text: str):
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    def stem(self, text: str):
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

def main():
    print("Hi!")
    pp = Preprocessor(
        ["Hi, My name's Ayoub that is an Iranian name which each girl loves to be beside one",
          "One day, One morning shining in the ocean",
          "We were both young when I first saw you!",
          "I can be reached at www.Ayoubekalantary.ir b.t.w!"],
          "stopwords.txt")
    print(pp.preprocess())

if __name__ == "__main__":
    main()
