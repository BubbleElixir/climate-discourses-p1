import numpy as np
import hdbscan
import plotly.graph_objects as go
import textwrap
from plotly.colors import qualitative
import pickle
import gc
import tqdm

# Load the list of UMAP embeddings files
with open('./reduced_activist_chunk_embeddings/umap_embeddings_files.pkl', 'rb') as f:
    umap_embeddings_files = pickle.load(f)

# Initialize lists to store the data
embeddings_umap_list = []
ids = []
metadatas = []
documents = []

# Load the UMAP-transformed embeddings from files
print("Loading UMAP-transformed embeddings...")
for umap_file in tqdm.tqdm(umap_embeddings_files, desc="Loading UMAP Embeddings"):
    with open(umap_file, 'rb') as f:
        batch_data = pickle.load(f)
    embeddings_umap_list.append(batch_data['embeddings_umap'])
    ids.extend(batch_data['ids'])
    metadatas.extend(batch_data['metadatas'])
    documents.extend(batch_data['documents'])

    del batch_data
    gc.collect()

# Concatenate the embeddings
embeddings_umap = np.vstack(embeddings_umap_list)
del embeddings_umap_list  # Free memory
gc.collect()

print(f"Total number of data points loaded: {embeddings_umap.shape[0]}")

print("Performing clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=25,
    metric='euclidean',  # UMAP embeddings are in Euclidean space
    cluster_selection_method='leaf'
)

cluster_labels = clusterer.fit_predict(embeddings_umap)

print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")

# Update the metadatas with the new cluster labels
for idx, meta in enumerate(metadatas):
    meta['umap_cluster'] = int(cluster_labels[idx])  # Ensure the label is an integer

# Save the updated metadatas back to the pickle files
print("Saving updated metadata back to pickle files...")
current_idx = 0
for umap_file in tqdm.tqdm(umap_embeddings_files, desc="Updating Pickle Files"):
    with open(umap_file, 'rb') as f:
        batch_data = pickle.load(f)
    batch_size = len(batch_data['ids'])
    batch_metadatas = metadatas[current_idx:current_idx + batch_size]
    batch_data['metadatas'] = batch_metadatas
    current_idx += batch_size

    # Save the updated batch data
    with open(umap_file, 'wb') as f:
        pickle.dump(batch_data, f)

    del batch_data, batch_metadatas
    gc.collect()

# Prepare the color palette
unique_labels = np.unique(cluster_labels)
colors = qualitative.Plotly  # Or any other suitable palette
color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
if -1 in unique_labels:
    color_map[-1] = 'black'  # For noise or outliers

def format_content(content, width=100):
    return '<br>'.join(textwrap.wrap(content, width))

# Initialize Plotly figure
fig = go.Figure()

# Loop over each unique cluster label to create a separate trace for each cluster
print("Preparing visualization...")
for label in unique_labels:
    indices = [i for i, l in enumerate(cluster_labels) if l == label]
    print(f"Cluster {label} has {len(indices)} data points.")

    if len(indices) == 0:
        continue  # Skip clusters with no data points

    # Extract data for the current cluster
    cluster_embeddings = embeddings_umap[indices]
    cluster_metadatas = [metadatas[i] for i in indices]
    cluster_documents = [documents[i] for i in indices]

    # Prepare hover text
    hover_texts = []
    for idx in range(len(indices)):
        meta = cluster_metadatas[idx]
        idea_unit = cluster_documents[idx]
        source_id = meta.get('source_index_col', 'Unknown')
        idea_unit_index = meta.get('idea_unit_index', 'Unknown')
        hover_text = (
            f"Idea Unit Index: {idea_unit_index}<br>"
            f"Source Document ID: {source_id}<br>"
            f"Cluster: {label}<br>"
            f"Length: {len(idea_unit)} characters<br>"
            f"Content: {format_content(idea_unit[:900])}"
        )
        hover_texts.append(hover_text)

    # Add a trace for this cluster
    fig.add_trace(go.Scatter3d(
        x=cluster_embeddings[:, 0],
        y=cluster_embeddings[:, 1],
        z=cluster_embeddings[:, 2],
        mode='markers',
        marker=dict(size=2, color=color_map[label], opacity=0.7),
        text=hover_texts,
        hoverinfo='text',
        name=f'Cluster {label}'
    ))

# Update the layout to add titles and enable the legend
fig.update_layout(
    title="3D UMAP Clustering of Idea Units",
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    ),
    legend_title="Clusters",
    legend=dict(
        x=0.8,  # Adjust legend position if necessary
        y=0.9,
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
print("Saving visualization...")
fig.write_html('umap_3d_cluster_idea_units_CLEAN.html')
print("Visualization saved as 'umap_3d_cluster_idea_units_CLEAN.html'")
