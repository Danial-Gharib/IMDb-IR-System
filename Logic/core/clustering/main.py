import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from word_embedding.fasttext_data_loader import FastTextDataLoader
from word_embedding.fasttext_model import FastText
from clustering.dimension_reduction import DimensionReduction
from clustering.clustering_metrics import ClusteringMetrics
from clustering.clustering_utils import ClusteringUtils
from tqdm import tqdm
# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.

ft = FastText()
ft.load_model(path="fasttext_training/FastText_model.bin")
print("fasttext model loaded.")
ft_loader = FastTextDataLoader(file_path="fasttext_training/")
# X, y = ft_loader.create_train_data(save=True)
X, y = ft_loader.load_traindata()
print("training data loaded")
embeddings = []
for sentence in tqdm(X, desc="Embeddings are being gathered"):
    embeddings.append(ft.get_query_embedding(sentence))

embeddings = np.array(embeddings)
print("embeddings generated.")
# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.

pca = DimensionReduction()


pca.wandb_plot_explained_variance_by_components(embeddings, 'Clustering', "variance by components")
print("Variance By Components finished")
# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.


embeddings_tsne = pca.convert_to_2d_tsne(embeddings)
embeddings_tsne = pca.pca_reduce_dimension(embeddings=embeddings, n_components=2)
pca.wandb_plot_2d_tsne(embeddings, "Clustering", "2d_tsne")
print("2D_TSNE finished")
# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
clustering_utils = ClusteringUtils()
min_k = 2
max_k = 15
for k in tqdm(range(min_k, max_k), desc=f"Visualizing kmeans with different k values"):
    clustering_utils.visualize_kmeans_clustering_wandb(embeddings_tsne, k, "kmeans Clustering", f"kmeans with k = {k}")

clustering_utils.plot_kmeans_cluster_scores(embeddings_tsne, y, [k for k in range(min_k, max_k)], 'Clustering', 'Kmeans Scores')

clustering_utils.visualize_elbow_method_wcss(embeddings_tsne, [k for k in range(min_k, max_k)], 'Clustering', 'Elbow WCSS')

print("clustering kmeans finished")
## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.

clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings_tsne, "Hierarchical Clustering", "single", "Single")
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings_tsne, "Hierarchical Clustering", "average", "Average")
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings_tsne, "Hierarchical Clustering", "complete", "Complete")
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings_tsne, "Hierarchical Clustering", "ward", "Ward")

print("wandb visualize hierarchical plots finished")
# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.
clustering_metrics = ClusteringMetrics()


def evaluate_metric(method="kmeans", k=5, y=None):
    def print_results(emb, labels, y):
        print(f"silhouette score : {clustering_metrics.silhouette_score(emb, labels)}")
        print(f"purity : {clustering_metrics.purity_score(y, labels)}")
        print(f"adjusted rand score : {clustering_metrics.adjusted_rand_score(y, labels)}")
    print("***********************************************************")
    print(method)
    if method == "kmeans":
        centers, labels = clustering_utils.cluster_kmeans(embeddings_tsne, k)
    elif method == "h_single":
        labels = clustering_utils.cluster_hierarchical_single(embeddings_tsne)
    elif method == "h_complete":
        labels = clustering_utils.cluster_hierarchical_complete(embeddings_tsne)
    elif method == "h_average":
        labels = clustering_utils.cluster_hierarchical_average(embeddings_tsne)
    elif method == "h_ward":
        labels = clustering_utils.cluster_hierarchical_ward(embeddings_tsne)
    else:
        raise ValueError("Invalid method name input")

    print_results(embeddings_tsne, labels, y)
    print("***********************************************************")


methods = ["kmeans", "h_single", "h_average", "h_complete", "h_ward"]

for method in methods:
    evaluate_metric(method=method, k=5, y=y)


print("metrics finished")