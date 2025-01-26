import numpy as np
import pickle
import gc
import tqdm

# Load the list of UMAP embeddings files
with open('umap_embeddings_files.pkl', 'rb') as f:
    umap_embeddings_files = pickle.load(f)

# Initialize lists to store the data
documents = []
cluster_labels = []
embeddings_umap_list = []

# Load the UMAP embeddings, documents, and cluster labels
print("Loading UMAP embeddings and cluster labels...")
for umap_file in tqdm.tqdm(umap_embeddings_files, desc="Loading Data"):
    with open(umap_file, 'rb') as f:
        batch_data = pickle.load(f)
    
    # Append data to the lists
    embeddings_umap_list.append(batch_data['embeddings_umap'])  # UMAP embeddings (n_samples x 3)
    documents.extend(batch_data['documents'])
    # Extract cluster labels from metadatas
    cluster_labels.extend([meta['umap_cluster'] for meta in batch_data['metadatas']])
    
    # Clean up to free memory
    del batch_data
    gc.collect()

# Concatenate the UMAP embeddings
embeddings_umap = np.vstack(embeddings_umap_list)
del embeddings_umap_list  # Free memory
gc.collect()

print(f"Total number of data points loaded: {embeddings_umap.shape[0]}")

# Remove outlier clusters if needed (e.g., cluster_label == -1)
valid_indices = [i for i, label in enumerate(cluster_labels) if label != -1]

# Filter the data to exclude outliers
embeddings_umap = embeddings_umap[valid_indices]
documents = [documents[i] for i in valid_indices]
cluster_labels = [cluster_labels[i] for i in valid_indices]

# Custom UMAP model that uses precomputed embeddings
class PrecomputedUMAP:
    def __init__(self, embedding):
        self.embedding_ = embedding  # Precomputed UMAP embeddings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.embedding_

from bertopic.cluster import BaseCluster

# Custom clustering model that uses precomputed cluster labels
class PrecomputedClusters(BaseCluster):
    def __init__(self, labels):
        self.labels_ = np.array(labels)

    def fit(self, X):
        return self

    def fit_predict(self, X):
        return self.labels_

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare sub-models
umap_model = PrecomputedUMAP(embeddings_umap)
hdbscan_model = PrecomputedClusters(cluster_labels)
vectorizer_model = CountVectorizer(min_df=10, max_df=0.6, stop_words="english")

# Representation models with the embedding model
main_representation_model = KeyBERTInspired(top_n_words=15)
aspect_representation_model1 = PartOfSpeech("en_core_web_sm")
aspect_representation_model2 = [
    KeyBERTInspired(top_n_words=30),
    MaximalMarginalRelevance(diversity=0.8)
]

representation_model = {
    "Main": main_representation_model,
    "Aspect1": aspect_representation_model1,
    "Aspect2": aspect_representation_model2
}

# Initialize BERTopic with the embedding model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    calculate_probabilities=False,
    verbose=True
)

embeddings_list = []

# Define batch size for loading in chunks to manage memory efficiently
batch_size = 5000
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="activist_idea_units")
collection_size = collection.count()  # Get total items in collection

print("Loading high-dimensional embeddings from ChromaDB...")
for i in tqdm.tqdm(range(0, collection_size, batch_size), desc="Loading Embeddings"):
    results = collection.get(
        where={},  # Retrieve all documents
        limit=batch_size,
        offset=i,
        include=['embeddings']
    )

    # Get embeddings from the result
    batch_embeddings = results.get('embeddings', [])
    
    # Ensure each batch of embeddings is added as a NumPy array to `embeddings_list`
    if batch_embeddings is not None and len(batch_embeddings) > 0:
        embeddings_list.extend(np.array(batch_embeddings))  # Appending directly if embeddings are in list format

    del results, batch_embeddings
    gc.collect()

# Concatenate the high-dimensional embeddings
embeddings = np.vstack(embeddings_list)
del embeddings_list  # Free memory
gc.collect()

# Filter embeddings to exclude outliers, same as before
embeddings = embeddings[valid_indices]

# Fit the topic model using the high-dimensional embeddings
topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)

topic_info = topic_model.get_topic_info()
topic_info.to_csv("./BERTopic_chunk_cluster_topics.csv", index=False)

from scipy.cluster import hierarchy as sch

# Generate the topic hierarchy
linkage_function = lambda x: sch.linkage(x, method='ward', optimal_ordering=True)

hierarchical_topics = topic_model.hierarchical_topics(
    documents,
    linkage_function=linkage_function
)

# Visualize the topic hierarchy with custom labels
hierarchy_fig = topic_model.visualize_hierarchy(
    hierarchical_topics=hierarchical_topics,
    orientation='left',
    title="Activist Chunk Topic Hierarchy",
)

# Show and save the figure
hierarchy_fig.write_html("activist_chunk_topic_hierarchy.html")

#doc_hierarchy_fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)

