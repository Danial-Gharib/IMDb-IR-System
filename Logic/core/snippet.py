from preprocess import Preprocessor
class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        pp = Preprocessor([query], 'stopwords.txt', does_lemmatize=False, does_stem=False)
        preprocessed_query = pp.preprocess()[0]
        res = ' '.join(q_word for q_word in preprocessed_query)
        return res

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        preprocessed_query = self.remove_stop_words_from_query(query)
        doc_tokens = doc.split()
        query_tokens = preprocessed_query.split()

        for query_token in query_tokens:
            if query_token in doc_tokens:
                occurences = [i for i, token in enumerate(doc_tokens) if token == query_token]
                for index in occurences:
                    start_idx = max(0, index - self.number_of_words_on_each_side)
                    end_idx = min(len(doc_tokens), index + self.number_of_words_on_each_side)
                    snippet_words = doc_tokens[start_idx:end_idx+1]
                    snippet_words = [f'***{word}***' if word == query_token else word for word in snippet_words]
                    snippet = " ".join(snippet_words)
                    final_snippet += snippet + " ... "
            else:
                not_exist_words.append(query_token)
        return final_snippet, not_exist_words

def main():
    print("Hi")
    snippet = Snippet(2)
    doc = "The quick brown fox jumps over the lazy dog."
    query = "quick fox saviour"
    snippet, not_exist_words = snippet.find_snippet(doc, query)
    print("Snippet:", snippet)
    print("Words not in doc:", not_exist_words)

if __name__ == "__main__":
    main()