import json
class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(word) - k + 1):
            shingles.add(word[i:i+k])
        # TODO: Create shingle here

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        if not first_set and not second_set:
            return 0.0
        intersec = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersec / union if union != 0 else 0.0

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()
        
        # TODO: Create shingled words dictionary and word counter dictionary here.
        for document in all_documents:
            for word in document.split():
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                    word_counter[word] = 1
                else:
                    word_counter[word] += 1

        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        for candidate_word in self.all_shingled_words:
            score = self.jaccard_score(self.shingle_word(word), self.all_shingled_words[candidate_word])
            if len(top5_candidates) < 5:
                top5_candidates.append((score, candidate_word))
            else:
                min_score = min(top5_candidates, key=lambda x: x[0])
                if score > min_score[0]:
                    top5_candidates.remove(min_score)
                    top5_candidates.append((score, candidate_word))
        top5_candidates.sort(reverse=True)
        return [candidate_word for _, candidate_word in top5_candidates]
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        candidates = self.find_nearest_words(query)
        tf_max = max(self.word_counter.get(candidate, 1) for candidate in candidates[:5])
        for candidate in candidates[:5]:
            tf_normalizaed = self.word_counter.get(candidate, 1) / tf_max
            score = tf_normalizaed * self.jaccard_score(self.shingle_word(query), self.all_shingled_words[candidate])
            # print(f"Candidate : {candidate}, Score : {score}")
            if score > 0:
                final_result = candidate
                break 
        # TODO: Do spell correction here.

        return final_result
    
def main():
    print("Hi")
    with open ('IMDB_crawled.json', 'r') as imdb_file:
        imdb_data = json.load(imdb_file)
    docs = [' '.join(movie['summaries']) for movie in imdb_data if movie['summaries'] and movie['summaries'] != 'not-present']
    spell_corr = SpellCorrection(docs)
    some_wrong_words = ["amazin", "perfekt", "woorm", "discusion", "chiken"]
    for ww in some_wrong_words:
        print(f"wrong word : {ww} ||| correct word : {spell_corr.spell_check(ww)}")
        print("_____________________________")
if __name__ == "__main__":
    main()