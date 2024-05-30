import time
import os
import json
import copy

## importing preprocessor
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
###
from indexer.indexes_enum import Indexes
from utility import Preprocessor

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        #         TODO
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            for star in doc.get('stars', []):
                if star not in current_index:
                    current_index[star] = {}
                current_index[star][doc['id']] = doc['stars'].count(star)
        return current_index
        pass

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            for genre in doc.get('genres', []):
                if genre not in current_index:
                    current_index[genre] = {}
                current_index[genre][doc['id']] = doc['genres'].count(genre)
        return current_index
        pass

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        #         TODO
        for doc in self.preprocessed_documents:
            for word in doc.get('summaries', []):
                if word not in current_index:
                    current_index[word] = {}
                current_index[word][doc['id']] = doc['summaries'].count(word)
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            #         TODO
            return list(self.index[index_type][word].keys())
            pass
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO
        self.index[Indexes.DOCUMENTS.value][document['id']] = document
        for star in document.get('stars', []):
            if star not in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star] = {}
            self.index[Indexes.STARS.value][star][document['id']] = document['stars'].count(star)
        for genre in document.get('genres', []):
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            self.index[Indexes.GENRES.value][genre][document['id']] = document['genres'].count(genre)
        for word in document.get('summaries', []):
            if word not in self.index[Indexes.SUMMARIES.value]:
                self.index[Indexes.SUMMARIES.value][word] = {}
            self.index[Indexes.SUMMARIES.value][word][document['id']] = document['summaries'].count(word)
        return        
        pass

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        if document_id in self.index[Indexes.DOCUMENTS.value]:
            del self.index[Indexes.DOCUMENTS.value][document_id]
        for star_index in self.index[Indexes.STARS.value].values():
            if document_id in star_index:
                del star_index[document_id]
        for genre_index in self.index[Indexes.GENRES.value].values():
            if document_id in genre_index:
                del genre_index[document_id]
        for summary_index in self.index[Indexes.SUMMARIES.value].values():
            if document_id in summary_index:
                del summary_index[document_id]
        #         TODO
        pass

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return
        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        # TODO
        with open(os.path.join(path, f'{index_name}_index.json'), 'w') as f:
            json.dump(self.index[index_name], f)
        return
        pass

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        for index_type in Indexes:
            file_path = os.path.join(path, f'{index_type.value}_index.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.index[index_type.value] = json.load(f)
        return    
        #         TODO
        pass

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break
        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
def main():
    curr_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(curr_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    grandgrand_dir = os.path.dirname(grandparent_dir)
    with open (os.path.join(grandgrand_dir, 'IMDB_crawled_standard.json'), 'r') as f:
        imdb_data = json.load(f)
    ###############
    ids = []
    summareis = []
    stars = []
    genres = []
    titles = []
    for movie in imdb_data:
        if movie['summaries'] is not None and movie['stars'] is not None and movie['genres'] is not None:
            ids.append(movie['id'])
            summareis.append(' '.join(movie['summaries']))
            stars.append(' '.join(movie['stars']))
            genres.append(' '.join(movie['genres']))
            titles.append(movie['title'])


    print("Start preprocessing")
    pp_sum = Preprocessor(summareis, stopwords_path='stopwords.txt', does_lemmatize=False)
    preprocessed_summaries = pp_sum.preprocess()
    pp_stars = Preprocessor(stars, stopwords_path='stopwords.txt', does_lemmatize=False, does_stem=False)
    preprocessed_stars = pp_stars.preprocess()
    pp_genres = Preprocessor(genres, stopwords_path='stopwords.txt', does_lemmatize=False, does_stem=False)
    preprocessed_genres = pp_genres.preprocess()
    print("preprocessing finished")
    ##############
    preprocessed_docs = []
    for movie_idx in range(len(ids)):
        movie = {
            'id':ids[movie_idx],
            'title': titles[movie_idx],
            'summaries':preprocessed_summaries[movie_idx],
            'stars':preprocessed_stars[movie_idx],
            'genres':preprocessed_genres[movie_idx]
        }
        preprocessed_docs.append(movie)
    ##### now testing the indexes :
    indexer = Index(preprocessed_documents=preprocessed_docs)

    indexer.check_add_remove_is_correct()

    indexer.check_if_indexing_is_good('summaries', 'perfect')


    indexer.store_index('indexes_standard', 'documents')
    indexer.store_index('indexes_standard', 'stars')
    indexer.store_index('indexes_standard', 'summaries')
    indexer.store_index('indexes_standard', 'genres')
    ### check if index loaded correctly for different index types 

    print(indexer.check_if_index_loaded_correctly('documents', indexer.index['documents']))
    print(indexer.check_if_index_loaded_correctly('stars', indexer.index['stars']))
    print(indexer.check_if_index_loaded_correctly('genres', indexer.index['genres']))
    print(indexer.check_if_index_loaded_correctly('summaries', indexer.index['summaries']))

if __name__ == "__main__":
    main()