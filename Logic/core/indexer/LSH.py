import numpy as np
import itertools
import random
import json
from utility.preprocess import Preprocessor
class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        words = document.split()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        return shingles
    
    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # TODO
        
        shingle_set = set()
        for document in self.documents:
            shingle_set.update(self.shingle_document(document))
        
        shingle_to_index = {shingle: i for i, shingle in enumerate(shingle_set)}

        num_shingles = len(shingle_set)
        num_docs = len(self.documents)
        characteristic_matrix = np.zeros((num_shingles, num_docs), dtype=int)

        for doc_idx, document in enumerate(self.documents):
            doc_shingles = self.shingle_document(document)
            for shingle in doc_shingles:
                characteristic_matrix[shingle_to_index[shingle], doc_idx] = 1
        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        # TODO
        characteristic_matrix = self.build_characteristic_matrix()
        num_shingles, num_docs = characteristic_matrix.shape
        signature_matrix = np.full((self.num_hashes, num_docs), np.inf)

        permutations = [np.random.permutation(num_shingles) for _ in range(self.num_hashes)]
        boolean_cm = characteristic_matrix.astype(bool)
        for perm_idx, permutation in enumerate(permutations):
            min_indices = np.argmax(boolean_cm[permutation], axis=0)
            signature_matrix[perm_idx] = min_indices

        return signature_matrix
    
    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        num_hashes, num_docs = signature.shape
        # bands = int(num_hashes / rows_per_band)
        buckets = {}
        for band_idx in range(bands):
            band_s = band_idx * rows_per_band
            band_e = (band_idx + 1) * rows_per_band
            if band_e > num_hashes:
                band_e = num_hashes
            for doc_idx in range(num_docs):
                band_hash = hash(tuple(signature[band_s:band_e, doc_idx]))
                if band_hash in buckets:
                    buckets[band_hash].append(doc_idx)
                else:
                    buckets[band_hash] = [doc_idx]
        return buckets
    

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        signature = self.min_hash_signature()
        buckets = self.lsh_buckets(signature)
        print("DONE")
        return buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        # TODO
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0
        pass

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0
        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

def main():
    print("Hi")
    with open ('IMDB_crawled.json', 'r') as imdb_file:
        imdb_data = json.load(imdb_file)
    real_movie_docs = [' '.join(movie['summaries']) for movie in imdb_data if movie['summaries'] and movie['summaries'] != 'not-present']
    with open('Logic/core/LSHFakeData.json', 'r') as fake_file:
        fake_data = json.load(fake_file)
    fake_movie_docs = [' '.join(movie['summaries']) for movie in fake_data]
    all_docs =  real_movie_docs + fake_movie_docs
    ####
    lsh = MinHashLSH(documents=all_docs, num_hashes=100)
    buckets = lsh.perform_lsh()
    print(f"Number of buckets : {len(buckets)}")
    lsh.jaccard_similarity_test(buckets, all_docs)
if __name__ == '__main__':
    main()