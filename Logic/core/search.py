import json
import numpy as np
from preprocess import Preprocessor
from scorer import Scorer
from indexer.indexes_enum import Indexes, Index_types
from indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = 'indexes/'
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES)
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED)
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH)
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA)

    def search(self, query, method, weights, safe_ranking = True, max_results=10):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0]

        scores = {}
        if safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)
        
        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        # TODO
        for index_type, weight in weights.items():
            for doc_id, score in scores[index_type].items():
                if doc_id not in final_scores:
                    final_scores[doc_id] = 0
                final_scores[doc_id] += score * weight
        return
        pass

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for field in weights:
            for tier in ["first_tier", "second_tier", "third_tier"]:
                #TODO
                scorer = Scorer(
                    self.tiered_index[field].index[tier],
                    self.metadata_index.index['document_count']
                )
                if method == 'OkapiBM25':
                    score = scorer.compute_socres_with_okapi_bm25(
                        query,
                        self.metadata_index.index['averge_document_length'][field.value],
                        self.document_lengths_index[field].index
                    )
                else:
                    score = scorer.compute_scores_with_vector_space_model(
                        query, method
                    )
                if field in scores:
                    scores[field] = self.merge_scores(scores[field], score)
                else:
                    scores[field] = score
                if len(scores[field]) >= max_results:
                    break
                

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """

        for field in weights:
            #TODO
            index_reader = self.document_indexes[field]
            index = index_reader.index
            scorer = Scorer(index, self.metadata_index.index['document_count'])
            if method == 'OkapiBM25':
                scores[field] = scorer.compute_socres_with_okapi_bm25(query, 
                                self.metadata_index.index['averge_document_length'][field.value],
                                self.document_lengths_index[field].index
                                )
            else:
                scores[field] = scorer.compute_scores_with_vector_space_model(
                    query, method
                    )
        return
        

    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        merged_scores = {}
        for doc_id in scores1:
            merged_scores[doc_id] = scores1[doc_id] + scores2.get(doc_id, 0)
        for doc_id in scores2:
            if doc_id not in merged_scores:
                merged_scores[doc_id] = scores2[doc_id]

        return merged_scores

        #TODO


if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "spiderman in wonderland"
    method = "lnc.ltc"
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights, safe_ranking=True)

    print(result)
