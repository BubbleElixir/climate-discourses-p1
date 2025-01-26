import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

# Check for command line arguments for ngram_range
if len(sys.argv) != 3:
    print("Usage: python script_name.py min_ngram max_ngram")
    sys.exit(1)

min_ngram, max_ngram = int(sys.argv[1]), int(sys.argv[2])

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to extract top terms for each cluster
def extract_top_terms(tfidf_matrix, feature_names, df, top_n=15):
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    df_tfidf['cluster'] = df['tmp_clusters']

    top_terms = {}
    for cluster in sorted(set(df['tmp_clusters'])):
        if cluster != -1:  # excluding noise
            cluster_data = df_tfidf[df_tfidf['cluster'] == cluster]
            mean_scores = cluster_data.mean(axis=0)
            top_indices = np.argsort(mean_scores)[-top_n:]
            top_indices = top_indices[top_indices < len(feature_names)]
            top_terms[cluster] = [(feature_names[i], mean_scores[i]) for i in top_indices]
    return top_terms

# Load data
df = load_data('clean_final_updated_activist_embeddings_tmp_clusters_12.csv')

# Select the appropriate content column
df['selected_content'] = df.apply(lambda x: x['clean_content'] if pd.notna(x['clean_content']) else x['content'], axis=1)
tfidf_vectorizer = TfidfVectorizer(max_features=4999, stop_words='english', ngram_range=(min_ngram, max_ngram))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['selected_content'])
feature_names = tfidf_vectorizer.get_feature_names_out()

top_terms_per_cluster = extract_top_terms(tfidf_matrix, feature_names, df)

# Generate output file name based on ngram range
output_file = f"cluster_terms_{min_ngram}_{max_ngram}.txt"

fig = go.Figure()
with open(output_file, 'w') as file:
    for cluster, terms in top_terms_per_cluster.items():
        file.write(f"\nCluster {cluster}:\n")
        for term, score in terms:
            file.write(f"{term}: {score}\n")
    terms, scores = zip(*terms)
    x_values = [cluster] * len(terms)
    text_sizes = [10 + score * 10 for score in scores]
    for x, y, term, text_size in zip(x_values, scores, terms, text_sizes):
        fig.add_trace(go.Scatter(
            x=[x], 
            y=[y], 
            mode='text',
            text=[term],
            textposition='top center',
            textfont=dict(size=text_size),
            showlegend=False
        ))

fig.update_layout(
    title="Top 15 TF-IDF Word Terms by Cluster",
    xaxis_title="Cluster",
    yaxis_title="TF-IDF Score",
    xaxis=dict(tickmode='array', tickvals=list(top_terms_per_cluster.keys())),
    width=4800,
    height=1600,
    yaxis=dict(type='log', autorange=True),
    margin=dict(l=0, r=0, t=10, b=0)
)
fig.write_html(f'top_{min_ngram}_{max_ngram}_grams_per_cluster.html')

print(f"Interactive visualization of top TF-IDF terms per cluster saved as HTML for n-grams {min_ngram} to {max_ngram}.")
