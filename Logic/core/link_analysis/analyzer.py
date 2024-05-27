from graph import LinkGraph
import json
# from ..indexer.indexes_enum import Indexes
# from ..indexer.index_reader import Index_reader
import random


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie_id = movie['id']
            self.graph.add_node(movie_id)
            self.authorities.append(movie_id)
            if movie['stars'] is not None:
                for star in movie['stars']:
                    if star not in self.hubs:
                        self.graph.add_node(star)
                        self.hubs.append(star)
                    self.graph.add_edge(star, movie_id)    
        

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """


        for movie in corpus:
            movie_id = movie['id']
            stars = movie['stars'] if movie['stars'] is not None else []
            for star in stars:
                if star in self.hubs and movie_id not in self.authorities:
                    self.authorities.append(movie_id)
                    self.graph.add_node(movie_id)
                    self.graph.add_edge(star, movie_id)

            

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        hub_scores = {node :1.0 for node in self.hubs}
        auth_scores = {node: 1.0 for node in self.authorities}

        for _ in range(num_iteration):
            update_auth_scores = {}
            update_hub_scores = {}
            for node in auth_scores:
                new_value = sum(hub_scores[pred] for pred in self.graph.get_predecessors(node))
                update_auth_scores[node] = new_value
            for node in hub_scores:
                new_value = sum(auth_scores[succ] for succ in self.graph.get_successors(node))
                update_hub_scores[node] = new_value
            norm_auth = sum(update_auth_scores.values())
            norm_hubs = sum(update_hub_scores.values())

            if norm_auth > 0:
                for node in update_auth_scores:
                    update_auth_scores[node] /= norm_auth
            if norm_hubs > 0:
                for node in update_hub_scores:
                    update_hub_scores[node] /= norm_hubs
            auth_scores = dict(update_auth_scores)
            hub_scores = dict(update_hub_scores)


        top_authorities = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)[:max_result]
        top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:max_result]
        # print(top_authorities)
        # print(top_hubs)
        top_movies = [node for node, _ in top_authorities if node in self.authorities]
        top_actors = [node for node, _ in top_hubs if node in self.hubs]
        #TODO

        return  top_actors, top_movies

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = []
    with open ('IMDB_crawled_standard.json') as f:
        corpus = json.load(f)
    print("corpus loaded")
    root_set = random.sample(corpus, 500)
    print("root_set initialized")
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    print("graph expanded")
    print("-----------------------------------------------------")
    actors, movies = analyzer.hits(num_iteration=10, max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
