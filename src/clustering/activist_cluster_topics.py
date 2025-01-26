from bertopic import BERTopic
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from cuml.manifold import UMAP
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster import hierarchy as sch

df = pd.read_csv("./FINAL_CLEAN_embeddings_activist_4.csv")
df_new_embeddings = pd.read_csv("./FINAL_CLEAN_embeddings_activist_3.csv")
df = df[df['tmp_clusters'] != -1]

# Combine content columns
df['final_content'] = df['clean_content'].combine_first(df['content'])
df['embedding'] = df_new_embeddings['embedding']
df['final_content'] = df['clean_content'].combine_first(df['content'])
df['embedding'] = df['embedding'].apply(eval)  # Convert from string to list
df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)
# model.to(device)

# Extract vocab to be used in BERTopic
vocab = collections.Counter()
tokenizer = CountVectorizer().build_tokenizer()

# Get all documents and embeddings
documents = df['final_content'].tolist()
embeddings = np.vstack(df['embedding'].values)
clusters = df['tmp_clusters'].tolist()

# Use cuML's UMAP to precompute reduced embeddings
umap_model_cuml = UMAP(
    n_neighbors=15,
    n_components=2,
    metric='cosine',
)

reduced_embeddings = umap_model_cuml.fit_transform(embeddings)

# Define the custom Dimensionality class with reduced embeddings
class Dimensionality:
    """ Use this for pre-calculated reduced embeddings """
    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.reduced_embeddings

# Prepare sub-models
umap_model = Dimensionality(reduced_embeddings)
hdbscan_model = BaseCluster() 
vectorizer_model = CountVectorizer(stop_words="english")
#representation_model = KeyBERTInspired()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

main_representation_model = KeyBERTInspired(top_n_words=15)
aspect_representation_model1 = PartOfSpeech("en_core_web_sm")
aspect_representation_model2 = [KeyBERTInspired(top_n_words=30), 
                                MaximalMarginalRelevance(diversity=0.8)]

representation_model = {
   "Main": main_representation_model,
   "Aspect1":  aspect_representation_model1,
   "Aspect2":  aspect_representation_model2 
}

vectorizer_model = CountVectorizer(min_df=10, max_df=0.6, stop_words = "english")


# Initialize and fit BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    top_n_words=10,
    verbose=True
)

topics, probabilities = topic_model.fit_transform(
    documents, embeddings=embeddings, y=clusters
)

topic_model.save("./activist_topic_model", serialization="safetensors", save_ctfidf=True, save_embedding_model="dunzhang/stella_en_400M_v5")

# Save topic information
topic_info = topic_model.get_topic_info()
topic_info.to_csv("./BERTopic_cluster_topics.csv", index=False)

# Add topics to DataFrame
df['topic'] = topics

# --- Generate the Topic Hierarchy ---

# # --- Reduce Topics to Higher-Level Groups ---
# new_topic_model = topic_model.reduce_topics(documents, nr_topics=80)

# # Get the mapping from original topics to higher-level topics
# mapping = new_topic_model.topic_mapper_.get_mappings()

# # Assign higher-level topics to documents
# df['higher_level_topic'] = df['topic'].map(mapping)

# # Get labels for higher-level topics
# new_topic_info = new_topic_model.get_topic_info()
# new_topic_info.to_csv("./BERTopic_higher_level_topics.csv", index=False)

# # Save the DataFrame with higher-level topics
# df.to_csv("./FINAL_CLEAN_with_higher_level_topics.csv", index=False)

linkage_function = lambda x: sch.linkage(x, method='ward', optimal_ordering=True)

hierarchical_topics = topic_model.hierarchical_topics(
    documents,
    linkage_function=linkage_function
)

# --- Prepare Custom Labels with Both Original and Higher-Level Topics ---

# # Get topic information from the original topic model
# original_topic_info = topic_model.get_topic_info()
# original_topic_labels = {}
# for index, row in original_topic_info.iterrows():
#     topic_id = row['Topic']
#     if topic_id == -1:
#         continue  # Skip outlier topic
#     label = row['Name']
#     original_topic_labels[topic_id] = label

# # Get labels for higher-level topics from the reduced topic model
# higher_level_topic_info = new_topic_model.get_topic_info()
# higher_level_labels = {}
# for index, row in higher_level_topic_info.iterrows():
#     topic_id = row['Topic']
#     if topic_id == -1:
#         continue
#     label = row['Name']
#     higher_level_labels[topic_id] = label

# # Shorten labels for readability
# def shorten_label(label, max_words=3):
#     words = label.split(', ')
#     short_label = ', '.join(words[:max_words])
#     return short_label

# for topic_id in original_topic_labels:
#     original_topic_labels[topic_id] = shorten_label(original_topic_labels[topic_id], max_words=3)
# for topic_id in higher_level_labels:
#     higher_level_labels[topic_id] = shorten_label(higher_level_labels[topic_id], max_words=3)

# # Create combined labels
# combined_labels = {}
# for topic_id, original_label in original_topic_labels.items():
#     higher_topic_id = mapping.get(topic_id, topic_id)
#     higher_label = higher_level_labels.get(higher_topic_id, f"Topic {higher_topic_id}")
#     combined_label = f"{original_label} ({higher_label})"
#     combined_labels[topic_id] = combined_label
# Define transportation-related query terms

# queries = ["motor", "vehicle", "transport", "traffic", "train", "bus", "car", "road", "cycle","airplane"]

# # Find topics similar to each query and deduplicate
# transportation_topics = set()
# for query in queries:
#     similar_topics, similarity = topic_model.find_topics(query, top_n=10)
#     transportation_topics.update(similar_topics)

# # Retrieve the transportation-related topics
# transportation_topics = list(transportation_topics)
# transportation_data = {topic: topic_model.get_topic(topic) for topic in transportation_topics}

# # Print or save the transportation-related topics
# for topic, words in transportation_data.items():
#     print(f"Topic {topic}: {words}")

# --- Visualize the Topic Hierarchy with Custom Labels ---
hierarchy_fig = topic_model.visualize_hierarchy(
    hierarchical_topics=hierarchical_topics,
    orientation='left',
    title="Activist Topic Hierarchy",
)

doc_hierarchy_fig = topic_model.visualize_hierarchical_documents(documents, hierarchical_topics, reduced_embeddings=reduced_embeddings)

hierarchy_fig.write_html("topic_hierarchy_with_combined_labels.html")
doc_hierarchy_fig.write_html("doc_topic_hierarchy.html")