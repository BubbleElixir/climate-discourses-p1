import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, space_eval
from sklearn.metrics import pairwise_distances 
import umap
import hdbscan

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(eval)  # Convert from string to list
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))  # Convert to float32 for normalization
    return df

def quantize_embeddings(embeddings, bits=16):
    max_val = 2**bits - 1
    quantized_embeddings = np.round(((embeddings + 1) / 2) * max_val).astype(np.uint16)
    return quantized_embeddings

def compute_cosine_distances(embeddings):
    # Scale quantized embeddings back to [-1, 1] range
    embeddings_float = (embeddings.astype(np.float32) / (2**16 - 1)) * 2 - 1
    return pairwise_distances(embeddings_float, metric='cosine')

def reduce_dimensions(embeddings, n_components=50, n_neighbors=15):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, metric='cosine')
    return reducer.fit_transform(embeddings)

def perform_clustering(embeddings, min_cluster_size=15, min_samples=1):
    # Directly use embeddings for clustering to enable MST
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        allow_single_cluster=True, 
        gen_min_span_tree=True
    )
    # Use reduced embeddings for clustering
    clusterer.fit(embeddings)
    return clusterer

def objective(params, embeddings, alpha=0.1):
    reduced_embeddings = reduce_dimensions(
        embeddings, 
        n_components=int(params['n_components']), 
        n_neighbors=int(params['n_neighbors'])
    )
    clusters = perform_clustering(
        reduced_embeddings, 
        min_cluster_size=int(params['min_cluster_size']), 
        min_samples=int(params['min_samples'])
    )
    
    # Current metric
    validity_score = clusters.relative_validity_
    
    # Noise ratio
    noise_ratio = np.sum(clusters.labels_ == -1) / len(clusters.labels_)
    
    # Combine them into a single score to minimize
    score = -validity_score + alpha * noise_ratio  # fmin wants to minimize this

    return score


def bayesian_search(embeddings, space, max_evals=100):
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, alpha=1.0)
    best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params = space_eval(space, best)
    print('Best parameters:', best_params)
    
    # Generate final clusters using best parameters
    best_reduced = reduce_dimensions(embeddings, n_components=int(best_params['n_components']), n_neighbors=int(best_params['n_neighbors']))
    best_clusters = perform_clustering(best_reduced, min_cluster_size=int(best_params['min_cluster_size']), min_samples=int(best_params['min_samples']))
    
    return best_params, best_clusters, trials

# Load data
df = load_data('../final_embeddings_contrarian_1.csv')
embeddings = np.vstack(df['embedding'].values)
quantized_embeddings = quantize_embeddings(embeddings)
reduced_embeddings = reduce_dimensions(quantized_embeddings)

# Define the parameter space
space = {
    'n_neighbors': hp.quniform('n_neighbors', 10, 60, 5),
    'n_components': hp.quniform('n_components', 50, 200, 10),
    'min_cluster_size': hp.quniform('min_cluster_size', 5, 60, 5),
    'min_samples': hp.quniform('min_samples', 1, 20, 1),
}

# Perform Bayesian optimization
best_params, best_clusters, trials = bayesian_search(reduced_embeddings, space)