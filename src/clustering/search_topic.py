import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to clear GPU memory
def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# Configure 4-bit quantization using BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True  # Enable 4-bit precision
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')

# Load the model with 4-bit quantization
model = AutoModel.from_pretrained(
    'Salesforce/SFR-Embedding-2_R',
    quantization_config=quantization_config,
    device_map="auto"  # Automatically map layers to available devices
)
model.eval()

# Clear GPU memory after loading the model
clear_gpu_memory()

# Function to generate embeddings (used during both training and inference)
def generate_embeddings(texts, tokenizer, model):
    batch_dict = tokenizer(
        texts, max_length=1024, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, -1, :]  # Use the last token's embedding
        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    return embeddings.cpu().numpy()

# Function to generate query embedding
def generate_query_embedding(query, tokenizer, model):
    batch_dict = tokenizer(
        query, max_length=1024, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = outputs.last_hidden_state[:, -1, :]  # Use the last token's embedding
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().numpy()

# Load the BERTopic model
topic_model = BERTopic.load("./activist_topic_model", embedding_model=model)

# Clear GPU memory again after loading BERTopic model
clear_gpu_memory()

# List of original documents used in training
df = pd.read_csv("./FINAL_CLEAN_embeddings_activist_4.csv")
df_new_embeddings = pd.read_csv("./FINAL_CLEAN_embeddings_activist_3.csv")
df = df[df['tmp_clusters'] != -1]

# Combine content columns
df['final_content'] = df['clean_content'].combine_first(df['content'])
df['embedding'] = df_new_embeddings['embedding']
df['final_content'] = df['clean_content'].combine_first(df['content'])
df['embedding'] = df['embedding'].apply(eval)  # Convert from string to list
df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))

original_docs = df['final_content'].tolist()

# Retrieve document information
document_info = topic_model.get_document_info(original_docs)

# Function to search for topics using a query embedding
def search_topics(query, topic_model, tokenizer, model, top_n=10):
    # Generate query embedding
    query_embedding = generate_query_embedding(query, tokenizer, model)

    # Debug: Check dimensions
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Topic embeddings shape: {topic_model.topic_embeddings_.shape}")

    # Directly compute cosine similarity
    similarities = cosine_similarity(query_embedding, topic_model.topic_embeddings_)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]  # Get indices of top topics

    # Display results
    print("Top Similar Topics:")
    for topic_id in top_indices:
        sim_score = similarities[0][topic_id]
        topic_words = topic_model.get_topic(topic_id)
        print(f"Topic {topic_id} (Similarity: {sim_score:.4f}): {topic_words}")

    # Clear GPU memory after processing
    clear_gpu_memory()

# Example query
#query = "transportation, cars, airplane, ships, cruises, traffic, bicycle, transit, buses"  # Replace with your search term
#search_topics(query, topic_model, tokenizer, model, top_n=15)

import plotly.graph_objects as go
import numpy as np

# Function to visualize and save proportions of overlapping climate topics
def save_climate_topics_normalized_plotly(topic_queries, topic_model, tokenizer, model, document_info):
    # Map to store topic IDs for each category
    topic_categories = {}
    similarity_threshold = 0.55  # Threshold for topic similarity

    # Identify topic IDs for each category based on queries
    for category, query in topic_queries.items():
        query_embedding = generate_query_embedding(query, tokenizer, model)
        similarities = cosine_similarity(query_embedding, topic_model.topic_embeddings_)
        top_indices = np.argsort(similarities[0])[::-1]  # Rank topics by similarity
        related_topics = [topic_id for topic_id in top_indices if similarities[0][topic_id] > similarity_threshold]
        topic_categories[category] = related_topics
        print(f"Identified Topics for {category}: {related_topics}")

    # Initialize a matrix to track category assignments
    category_matrix = np.zeros((len(document_info), len(topic_queries)))

    # Fill the matrix: 1 if document matches a category, else 0
    for i, (category, topic_ids) in enumerate(topic_categories.items()):
        category_matrix[:, i] = document_info['Topic'].isin(topic_ids).astype(float)

    # Calculate the weight for each document in each category (adjust for overlaps)
    overlap_counts = category_matrix.sum(axis=1)  # Number of categories each document belongs to
    normalized_weights = np.divide(
        category_matrix,
        overlap_counts[:, None],
        out=np.zeros_like(category_matrix),  # Default to 0 where overlap_counts is 0
        where=overlap_counts[:, None] != 0  # Only divide where overlap_counts != 0
    )

    # Sum weights for each category to get normalized proportions
    category_sums = normalized_weights.sum(axis=0)
    total_weight = category_sums.sum()  # Ensure we sum to 100%
    proportions = category_sums / total_weight * 100

    # Prepare data for visualization
    labels = list(topic_queries.keys()) + ["Other Topics"]
    sizes = list(proportions) + [100 - proportions.sum()]

    # Create and save the pie chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent')])
    fig.update_layout(title='Proportions of Climate Change Topics (Normalized for Overlap)')
    fig.write_image("./climate_topics_normalized_pie_chart_2.png")
    print("Interactive pie chart saved as HTML at './climate_topics_normalized_pie_chart_2.html'")

# Define queries for climate change-related topics
topic_queries = {
    "Transportation": "EV, transportation, transit, transport, vehicles, buses, airplane, bicycles, emissions",
    "Clean Energy": "renewable energy, clean energy, solar power, wind turbines, electricity, energy policy",
    "Forestry": "deforestation, forest conservation, tree planting, reforestation",
    "Activism": "activism, protests, environmental advocacy, Greta Thunberg",
    "Policy": "policy, carbon tax, international agreements",
    "Fossil Fuels": "oil, coal, gas, fossil fuels"
}

# Save the visualization
save_climate_topics_normalized_plotly(topic_queries, topic_model, tokenizer, model, document_info)




