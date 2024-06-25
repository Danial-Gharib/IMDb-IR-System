import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            # TODO
            df = len(self.index.get(term, {}))
            idf = np.log(self.N / df)
            self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        query_tfs = {}
        for term in query:
            query_tfs[term] = query_tfs.get(term, 0) + 1
        return query_tfs
        #TODO

        # TODO

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        # TODO
        query_tfs = self.get_query_tfs(query)
        document_method, query_method = method.split('.')
        for doc_id in self.get_list_of_documents(query):
            scores[doc_id] = self.get_vector_space_model_score(
                query, query_tfs, doc_id, document_method, query_method)
        return scores
        pass

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        doc_tf_method, doc_idf_method, doc_norm_method = document_method[0], document_method[1], document_method[2]
        query_tf_method, query_idf_method, query_norm_method = query_method[0], query_method[1], query_method[2]
        q_v = list()
        d_v = list()
        query_terms = list(set(query))
        for term in query_terms:
            if term in self.index:
                q_v.append(self.calc_weight(query_tfs[term], self.get_idf(term), query_tf_method, query_idf_method))
                d_v.append(self.calc_weight(self.index[term].get(document_id, 0), self.get_idf(term)
                                            , doc_tf_method, doc_idf_method))
        if doc_norm_method == 'c':
            d_v = self.normalization(d_v)
        if query_norm_method == 'c':
            q_v = self.normalization(q_v)
        return np.dot(np.array(q_v), np.array(d_v))
    
    def calc_weight(self, tf, idf, tf_method, idf_method):
        res = 0
        if tf_method == 'n':
            res = tf
        elif tf_method == 'l':
            if tf != 0:
                res = np.log(tf) + 1
            else:
                res = 0
        if idf_method == 't':
            res *= idf
        return res
    def normalization(self, vector):
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        return list(vector / norm)
    
        

    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        for doc_id in self.get_list_of_documents(query):
            scores[doc_id] = self.get_okapi_bm25_score(query, doc_id, average_document_field_length, document_lengths)
        return scores
        # TODO
        pass

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        k1 = 1.2
        b = 0.75

        score = 0.0
        dl = document_lengths.get(document_id, 0)
        for term in query:
            if term not in self.index:
                continue
            df = len(self.index[term])
            okapi_idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf = self.index[term].get(document_id, 0)
            okapi_tf = (tf * (k1 + 1)) / (tf + k1*(1 - b + b*dl/average_document_field_length))
            score += okapi_idf * okapi_tf

        return score
        # TODO
        pass

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        for doc_id in self.get_list_of_documents(query):
            scores[doc_id] = self.compute_score_with_unigram_model(
                query, doc_id, smoothing_method, document_lengths, alpha, lamda
            )
        # TODO
        return scores
        pass

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """
        doc_lenghts = document_lengths.get(document_id, 0)
        total_terms_in_collection = sum(document_lengths.values())
        score = 0.0

        for term in query:
            term_freq = self.index.get(term, {}).get(document_id, 0)
            # print(term_freq)
            collection_term_freq = sum(self.index.get(term, {}).values())
            # print(collection_term_freq)
            # print(term)
            # print("_________________________")
            probability = 0.0
            if smoothing_method == 'naive':
                probability = term_freq / doc_lenghts if doc_lenghts > 0 else 0
            elif smoothing_method == 'bayes':
                collection_prob = collection_term_freq / total_terms_in_collection
                probability = (term_freq + alpha * collection_prob) / (doc_lenghts + alpha)
            elif smoothing_method == 'mixture':
                doc_prob = term_freq / doc_lenghts if doc_lenghts > 0 else 0
                collection_prob = collection_term_freq / total_terms_in_collection
                probability = lamda * doc_prob + (1 - lamda) * collection_prob
            else:
                raise ValueError("Unsopported smoothing method!")
            if probability > 0:
                score += np.log(probability)
        return score
        # TODO
        pass
