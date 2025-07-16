import numpy as np
import pandas as pd
import random

# Packages for HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import HDBSCAN
from joblib import Parallel, delayed
import umap
import umap.plot
import numpy as np

# Package to save models
import joblib

def get_embeddings(df):
    # Step 1: Initialize Smaller Model
    # embedder = SentenceTransformer("all-mpnet-base-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: Ensure Preprocessed Corpus has a Continuous Index
    preprocessed_corpus = df.reset_index(drop=True)

    # Convert to a list for parallel processing
    corpus_list = preprocessed_corpus.tolist()

    # Step 3: Batch Embedding Function
    def embed_batch(batch):
        return embedder.encode(batch, show_progress_bar=False)

    # Step 4: Generate Embeddings in Batches (Parallelized)
    def parallel_embedding(corpus, batch_size=512):
        embeddings = Parallel(n_jobs=1)(
            delayed(embed_batch)(corpus[i:i + batch_size])
            for i in range(0, len(corpus), batch_size)
        )
        return np.vstack(embeddings)

    # Encode the corpus in parallel
    batch_size = 512
    corpus_embeddings = parallel_embedding(corpus_list, batch_size=batch_size)

    return corpus_list, corpus_embeddings


def clusterize(corpus_list, corpus_embeddings,
               params,
               print_clusters=False):

    reducer = umap.UMAP(
        n_neighbors=params['umap_n_neighbors'],  # balance local vs global structure
        min_dist=params['umap_min_dist'],  # how tightly UMAP packs points together
        n_components=params['umap_n_components'],  # target dimensionality
        metric="cosine",  # distance metric on original space
        random_state=42,  # for reproducibility
    )

    # 3) Fit & transform
    reduced_embeddings = reducer.fit_transform(corpus_embeddings)
    reduced_embeddings = normalize(reduced_embeddings)

    # Step 6: Apply HDBSCAN Clustering
    # HDBSCAN automatically determines the number of clusters
    hdbscan_model = HDBSCAN(
        min_cluster_size=params['hdbscan_min_cluster_size'],
        min_samples=params['hdbscan_min_samples'],
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=params['hdbscan_cluster_selection_epsilon'],
        max_cluster_size = params['hdbscan_max_cluster_size']
    )
    cluster_assignment = hdbscan_model.fit_predict(reduced_embeddings)

    # Step 7: Analyze and Visualize Clusters
    # HDBSCAN assigns -1 to noise points
    num_clusters_found = len(set(cluster_assignment)) - (1 if -1 in cluster_assignment else 0)
    print(f"Number of clusters found: {num_clusters_found}")

    # Group sentences by cluster
    clustered_sentences = [[] for _ in range(num_clusters_found)]
    noise = []
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id != -1:  # Exclude noise points
            clustered_sentences[cluster_id].append(corpus_list[sentence_id])
        else:
            noise.append(corpus_list[sentence_id])

    # Print clusters
    if print_clusters:
        for i, cluster in enumerate(clustered_sentences):
            print(f"Cluster {i + 1}:")
            print(cluster)
            print("")

    if params['umap_n_components'] == 2:
        umap.plot.points(reducer, labels=cluster_assignment)

    return hdbscan_model, cluster_assignment, clustered_sentences, noise, reduced_embeddings


def get_noise_sample(noise, n_samples=100):
    noise_dict = {'samples': random.sample(noise, n_samples)}
    return pd.DataFrame.from_dict(noise_dict)


def get_cluster_summary(clustered_sentences):
    ## Extraction of a random sample of sentences to validation
    # Suppress duplicate sentences
    unique_clustered_sentences = [
        list(set(cluster)) for cluster in clustered_sentences
    ]

    # Create the DataFrame
    data = []
    for cluster_num, sentences in enumerate(unique_clustered_sentences):
        # Get the number of sentences in the cluster
        num_sentences = len(sentences)

        # Randomly sample 15 sentences (or fewer if the cluster has less than 10 sentences)
        sample_sentences = random.sample(sentences, min(15, num_sentences))

        # Append the cluster info to the data list
        data.append({
            "Cluster Number": cluster_num + 1,
            "Number of Sentences": num_sentences,
            "Sample Sentences": "; ".join(sample_sentences)
        })

    # Convert to DataFrame
    return pd.DataFrame(data)


def save_results(params, results, noise_df, cluster_summary_df, hdbscan_model, domain, cat):
    #faire une version qui enregistre chaque essai
    # import uuid

    # length = 8
    # random_string = str(uuid.uuid4()).replace('-', '')[:length]

    pd.DataFrame.from_dict(params, orient='index', columns=['valeur']).to_csv(f'../outputs/params_{domain}_{cat}.csv')
    pd.DataFrame.from_dict(results, orient='index', columns=['valeur']).to_csv(f'../outputs/results_{domain}_{cat}.csv')
    noise_df.to_csv(f'../outputs/noise_sample_{domain}_{cat}.csv')
    cluster_summary_df.to_csv(f'../outputs/clusters_hdbscan_{domain}_{cat}.csv', index=False)
    joblib.dump(hdbscan_model, f'../pickles/hdbscan_model_{domain}_{cat}.pkl')


# Function to subdivide clusters and create a new dataframe
def subdivide_clusters(clustered_sentences, cluster_assignment, reduced_embeddings, preprocessed_corpus, clusters_to_subdivide):
    # Create a list for the final combined clusters
    combined_clusters = []
    new_subclusters = []  # To hold subdivided clusters
    new_noise = []

    cluster_assignment_to_change_indices = []
    new_cluster_assignment = []

    # Add clusters that are not being subdivided to the final list
    for cluster_id, sentences in enumerate(clustered_sentences):
        if (cluster_id) not in clusters_to_subdivide:  # Adjust for 1-based indexing in `clusters_to_subdivide`
            combined_clusters.append({"cluster_id": cluster_id, "sentences": sentences})

    # Subdivide the specified clusters
    for cluster_index in clusters_to_subdivide.keys():
        # Adjust index for 0-based indexing (Python lists)
        cluster_id = cluster_index

        # Extract the embeddings and sentences for the current cluster
        indices = [i for i, cid in enumerate(cluster_assignment) if cid == cluster_index]

        cluster_assignment_to_change_indices.extend(indices)

        if len(indices) < 5:  # HDBSCAN needs at least a few points
            continue

        cluster_embeddings = reduced_embeddings[indices]
        cluster_sentences = [list(preprocessed_corpus)[i] for i in indices]

        subclusters_noise = []

        # Apply HDBSCAN to subdivide the cluster
        # Apply HDBSCAN to subdivide the cluster
        hdbscan_model = HDBSCAN(
            min_cluster_size=clusters_to_subdivide[cluster_index]['min_cluster_size'],  # Minimum cluster size
            min_samples=clusters_to_subdivide[cluster_index]['min_samples'],        # Minimum samples in a neighborhood for a core point
            metric='euclidean',   # Distance metric
            cluster_selection_epsilon=clusters_to_subdivide[cluster_index]['cluster_selection_epsilon'],  # Adjust for fine-grained clustering
            )
        hdbscan_labels = hdbscan_model.fit_predict(cluster_embeddings)

        new_cluster_assignment.extend([f'{cluster_index}_{label}' if label != -1 else label for label in hdbscan_labels])

        # Map each HDBSCAN cluster to the combined list
        for hdbscan_cluster_id in set(hdbscan_labels):
            if hdbscan_cluster_id == -1:  # Skip noise
                subclusters_noise = [cluster_sentences[i] for i, label in enumerate(hdbscan_labels) if label == -1]
                new_noise.extend(subclusters_noise)
            else:
                new_subclusters.append(
                {
                    "cluster_id": f"{cluster_index}-{hdbscan_cluster_id}",
                    "sentences": [cluster_sentences[i] for i, label in enumerate(hdbscan_labels) if label == hdbscan_cluster_id],
                }
            )

        if subclusters_noise:
            print(f'reclustering of cluster {cluster_index} adding {len(subclusters_noise)} items to noise')
        else:
            print(f'no noise generated by cluster {cluster_index}')

    # Append subdivided clusters to the remaining clusters
    combined_clusters.extend(new_subclusters)

    # Convert the combined clusters into a dataframe
    new_cluster_df = pd.DataFrame(combined_clusters)
    return new_cluster_df, new_noise, cluster_assignment_to_change_indices, new_cluster_assignment



def create_final_df(sentences_df, doi_col, ini_cluster_assignment, new_cluster_assignement, indices_to_update):
    sentences_df.insert(1, "doi", doi_col)
    sentences_df.insert(1, "cluster", ini_cluster_assignment)

    updates = dict(zip(indices_to_update, new_cluster_assignement))
    updated = [updates.get(i, val) for i, val in enumerate(ini_cluster_assignment)]

    sentences_df.insert(2, "cluster_2", updated)

    return sentences_df

















#############################################################
#############################################################
############ Test de reclustering du noise ?     ############
#############################################################
#############################################################
# cluster_id = -1

# combined_clusters = []
# new_subclusters = []

# # Extract the embeddings and sentences for the current cluster
# indices = [i for i, cid in enumerate(cluster_assignment) if cid == cluster_id]

# cluster_embeddings = corpus_embeddings_items[indices]
# cluster_sentences = [list(preprocessed_items)[i] for i in indices]


# hdbscan_model = HDBSCAN(
#     min_cluster_size=30,  # Minimum cluster size
#     min_samples=10,        # Minimum samples in a neighborhood for a core point
#     metric='euclidean',   # Distance metric
#     cluster_selection_epsilon=0.0,  # Adjust for fine-grained clustering
#     )
# hdbscan_labels = hdbscan_model.fit_predict(cluster_embeddings)

# for hdbscan_cluster_id in set(hdbscan_labels):
#     if hdbscan_cluster_id == -1:  # Skip noise
#         subclusters_noise = [cluster_sentences[i] for i, label in enumerate(hdbscan_labels) if label == -1]
#         new_noise.extend(subclusters_noise)
#     else:
#         new_subclusters.append(
#         {
#             "cluster_id": f"{cluster_index}-{hdbscan_cluster_id}",
#             "sentences": [cluster_sentences[i] for i, label in enumerate(hdbscan_labels) if label == hdbscan_cluster_id],
#         }
#     )
