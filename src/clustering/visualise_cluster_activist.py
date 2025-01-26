import numpy as np
import pandas as pd
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.colors import qualitative
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns

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
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, metric='cosine')
    return reducer.fit_transform(embeddings)

def perform_clustering(embeddings, min_cluster_size=15, min_samples=1):
    # Directly use embeddings for clustering to enable MST
    clusterer = HDBSCAN(
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

df = load_data('./FINAL_CLEAN_embeddings_activist_3.csv')
embeddings = np.vstack(df['embedding'].values)

# # Use the optimized parameters from Bayesian optimization
# reduced_embeddings = reduce_dimensions(embeddings, n_components=47, n_neighbors=28, random_state=43)
# clusters = perform_clustering(reduced_embeddings, min_cluster_size=13, min_samples=5)

quantized_embeddings = quantize_embeddings(embeddings)
reduced_embeddings = reduce_dimensions(quantized_embeddings, n_components=150, n_neighbors=25)
clusters = perform_clustering(reduced_embeddings, min_cluster_size=35, min_samples=1)

df['tmp_clusters'] = clusters.labels_
df.drop(["embedding"],axis=1)
df.to_csv("FINAL_CLEAN_embeddings_activist_10.csv", index=False)
plot_embeddings = reduce_dimensions(reduced_embeddings, n_components=3, n_neighbors=25)

# Prepare the color palette (using Plotly's qualitative palettes)
colors = qualitative.Plotly  # Or another qualitative palette such as D3, G10, T10, etc.
color_map = {label: colors[i % len(colors)] for i, label in enumerate(np.unique(clusters.labels_))}
color_map[-1] = 'black'

def format_content(content, width=100):
    return '<br>'.join(textwrap.wrap(content, width))

fig = go.Figure()

# Loop over each unique cluster label to create a separate trace for each cluster
unique_labels = set(clusters.labels_)
for label in unique_labels:
    # Get indices for data points in this cluster
    indices = [i for i, l in enumerate(clusters.labels_) if l == label]
    
    # Add a trace for this cluster
    fig.add_trace(go.Scatter3d(
        x=plot_embeddings[indices, 0],
        y=plot_embeddings[indices, 1],
        z=plot_embeddings[indices, 2],
        mode='markers',
        marker=dict(size=5, color=color_map[label], opacity=0.8),
        text = [f"Index: {df.index[i]}<br>Domain: {df.iloc[i]['domain']}<br>Title: {df.iloc[i]['cleaned_title'] if pd.notna(df.iloc[i]['cleaned_title']) else df.iloc[i]['title']}<br>Cluster: {label}<br>Length: {df.iloc[i]['length']}<br>Excerpt: {format_content(df.iloc[i]['clean_content'][:900] if pd.notna(df.iloc[i]['clean_content']) else df.iloc[i]['content'][:900])}" 
    for i in indices],
        hoverinfo='text',
        name=f'Cluster {label}'  # Assign a name for the legend based on the cluster label
    ))

# Update the layout to add titles and enable the legend
fig.update_layout(
    title="3D UMAP Clustering with HDBSCAN",
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    ),
    legend_title="Clusters",
    legend=dict(
        x=1,  # Adjust legend position if necessary
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    )
)

# Save the plot as an HTML file
fig.write_html('umap_3d_cluster_activist_final.html')
print("3D interactive cluster visualization saved as HTML.")
