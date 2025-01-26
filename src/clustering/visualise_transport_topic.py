from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from bertopic import BERTopic
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# Load sentiment analysis model and tokenizer
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model.eval()

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load the BERTopic model
topic_model = BERTopic.load("./activist_topic_model")

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
topic_info = topic_model.get_topic_info()
# # Filter Transportation Documents
# transportation_documents = document_info[document_info["Topic"] == transportation_topic_id]
# print(f"Number of Documents in Topic {transportation_topic_id}: {len(transportation_documents)}")

# Define text chunker
def split_into_chunks(text, chunk_size=512):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


# Analyze sentiment for a single topic
def analyze_sentiment_for_topic(documents, max_length=512):
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

    for doc in documents["Document"]:
        chunks = list(split_into_chunks(doc, chunk_size=max_length))
        for chunk in chunks:
            encoded_input = sentiment_tokenizer(
                chunk, truncation=True, max_length=max_length, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = sentiment_model(**encoded_input)
                sentiment = torch.argmax(outputs.logits).item()
                sentiments[label_mapping[sentiment]] += 1

    total = sum(sentiments.values())
    return {k: v / total for k, v in sentiments.items()}  # Normalize to proportions


# Selected transportation-related topics
transportation_topics = [127, 13, 107, 50, 79, 37, 100, 66, 31, 52]
clean_energy_topics: [67, 120, 57, 123, 25, 71, 75, 31, 64, 37]
forestry_topics: [24, 63, 4, 69, 38, 0, 100, 30, 85, 58]
activism_topics: [113, 58, 30, 100, 8, 39, 18, 103, 74, 126]
policy_topics: [114, 10, 6, 118, 23, 124, 106, 52, 65, 17]
fossil_fuel_topics: [119, 41, 29, 21, 54, 31, 42, 67, 25, 47]

# Filter documents for the selected topics
# selected_documents = document_info[document_info["Topic"].isin(selected_topics)]

# # Analyze sentiment for all selected topics
# def analyze_all_topic_sentiments(selected_topics, document_info):
#     topic_sentiments = {}
#     for topic_id in selected_topics:
#         topic_docs = document_info[document_info["Topic"] == topic_id]
#         topic_sentiments[topic_id] = analyze_sentiment_for_topic(topic_docs)
#     return topic_sentiments

# all_topic_sentiments = analyze_all_topic_sentiments(selected_topics, document_info)

# def extract_entities(documents, nlp):
#     entities = []
#     for doc in documents["Document"]:
#         spacy_doc = nlp(doc)
#         entities.extend([ent.text for ent in spacy_doc.ents if ent.label_ in {"ORG", "GPE", "PERSON", "PRODUCT", "LOC"}])
#     if not entities:
#         print("No entities found in the provided documents.")
#     return Counter(entities)

# # Extract entities for all selected topics
# def extract_entities_for_topics(selected_topics, document_info, nlp):
#     topic_entities = {}
#     for topic_id in selected_topics:
#         topic_docs = document_info[document_info["Topic"] == topic_id]
#         topic_entities[topic_id] = extract_entities(topic_docs, nlp)
#     return topic_entities

# all_topic_entities = extract_entities_for_topics(selected_topics, document_info, nlp)

# # 1. Sentiment Comparison Radar Plot
# def plot_sentiment_radar(all_topic_sentiments, selected_topics):
#     sentiments = ["Positive", "Neutral", "Negative"]
#     fig = go.Figure()

#     for topic_id in selected_topics:
#         topic_sentiments = [
#             all_topic_sentiments[topic_id].get(s, 0) for s in sentiments
#         ]
#         fig.add_trace(
#             go.Scatterpolar(
#                 r=topic_sentiments,
#                 theta=sentiments,
#                 fill="toself",
#                 name=f"Topic {topic_id}",
#             )
#         )

#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True)),
#         title="Sentiment Comparison Across Transportation-Related Topics",
#         showlegend=True,
#     )
#     fig.write_html("sentiment_comparison_radar.html")
#     print("Radar plot saved as 'sentiment_comparison_radar.html'")

# plot_sentiment_radar(all_topic_sentiments, selected_topics)

# # 2. Entity Mentions Bar Plot
# def plot_entity_mentions_bar(all_topic_entities):
#     combined_entity_counts = Counter()
#     for topic_entities in all_topic_entities.values():
#         combined_entity_counts.update(topic_entities)

#     if not combined_entity_counts:
#         print("No entities found to plot.")
#         return

#     # Extract top 20 entities
#     top_entities = combined_entity_counts.most_common(20)
#     if not top_entities:
#         print("No significant entities to display.")
#         return

#     entities, counts = zip(*top_entities)

#     # Plot using Plotly
#     fig = px.bar(
#         x=list(counts),
#         y=list(entities),
#         orientation="h",
#         title="Top Entities in Transportation-Related Topics",
#         labels={"x": "Mentions", "y": "Entities"},
#         template="plotly_white",
#     )
#     fig.update_layout(yaxis=dict(categoryorder="total ascending"))
#     fig.write_html("top_entities_transportation_topics.html")
#     print("Entity mentions bar plot saved as 'top_entities_transportation_topics.html'")

# plot_entity_mentions_bar(all_topic_entities)

# # 3. Word Cloud for Top Entities
# def plot_entity_wordcloud(all_topic_entities):
#     combined_entity_counts = Counter()
#     for topic_entities in all_topic_entities.values():
#         combined_entity_counts.update(topic_entities)

#     wordcloud = WordCloud(width=800, height=400, background_color="white")
#     wordcloud.generate_from_frequencies(dict(combined_entity_counts))

#     plt.figure(figsize=(12, 6))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.title("Word Cloud of Entities in Transportation Topics")
#     plt.tight_layout()
#     plt.savefig("entity_wordcloud_transportation_topics.png")
#     plt.close()

# plot_entity_wordcloud(all_topic_entities)


# def plot_keyword_wordcloud(topic_model, selected_topics, top_n=500):
#     """
#     Create a word cloud for the top keywords in selected topics.
#     """
#     from wordcloud import WordCloud

#     # Combine keywords from selected topics
#     combined_keywords = []
#     for topic_id in selected_topics:
#         combined_keywords.extend(topic_model.get_topic(topic_id))

#     # Filter for top N keywords
#     combined_keywords = sorted(combined_keywords, key=lambda x: x[1], reverse=True)[:top_n]
#     keyword_weights = {word: weight for word, weight in combined_keywords}

#     # Generate word cloud
#     wordcloud = WordCloud(width=800, height=400, background_color="white")
#     wordcloud.generate_from_frequencies(keyword_weights)

#     # Plot
#     plt.figure(figsize=(12, 6))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.title("Keyword Word Cloud", fontsize=16)
#     plt.tight_layout()
#     plt.savefig("keyword_wordcloud.png", dpi=150)
#     print("Word cloud saved as 'keyword_wordcloud.png'")

# # Call the function
# plot_keyword_wordcloud(topic_model, selected_topics)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

# Function to generate top bi-grams
def generate_bi_grams(docs, top_n=30):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")
    X = vectorizer.fit_transform(docs)
    word_counts = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    bi_grams = sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)[:top_n]
    return bi_grams

def generate_grouped_sunburst_chart(topic_model, document_info):
    # Define your grouped topics
    grouped_topics = {
        "Transportation": [127, 13, 107, 50, 79, 37, 100, 66, 31, 52],
        "Clean Energy": [67, 120, 57, 123, 25, 71, 75, 31, 64, 37],
        "Forestry": [24, 63, 4, 69, 38, 0, 100, 30, 85, 58],
        "Activism": [113, 58, 30, 100, 8, 39, 18, 103, 74, 126],
        "Policy": [114, 10, 6, 118, 23, 124, 106, 52, 65, 17],
        "Fossil Fuels": [119, 41, 29, 21, 54, 31, 42, 67, 25, 47],
    }

    # Prepare the data
    labels = []
    parents = []
    values = []

    # Amplification factor for sub-themes
    subtheme_amplification_factor = 2  # Adjust as needed

    # for group_name, topic_ids in grouped_topics.items():
    #     # Add the group as a parent
    #     labels.append(group_name)
    #     parents.append("")
    #     group_value = sum(len(document_info[document_info["Topic"] == topic_id]) for topic_id in topic_ids)
    #     values.append(group_value)

    #     # Aggregate sub-themes across all topics in this group
    #     subtheme_counter = Counter()
    #     for topic_id in topic_ids:
    #         sub_themes = topic_model.get_topic(topic_id)  # Extract sub-themes
    #         subtheme_counter.update({sub_theme: weight for sub_theme, weight in sub_themes})

    #     # Add aggregated sub-themes as children
    #     for sub_theme, weight in subtheme_counter.most_common():
    #         labels.append(sub_theme)
    #         parents.append(group_name)
    #         values.append(weight * subtheme_amplification_factor)  # Amplify sub-theme sizes

    for group_name, topic_ids in grouped_topics.items():
        # Add the group as a parent
        labels.append(group_name)
        parents.append("")
        group_value = sum(len(document_info[document_info["Topic"] == topic_id]) for topic_id in topic_ids)
        values.append(group_value)

        # Combine documents for all topics in this group
        group_docs = []
        for topic_id in topic_ids:
            topic_docs = document_info[document_info["Topic"] == topic_id]["Document"].tolist()
            group_docs.extend(topic_docs)

        # Generate bi-grams for the group's documents
        bi_grams = generate_bi_grams(group_docs, top_n=30)  # Adjust `top_n` as needed
        for bi_gram, count in bi_grams:
            labels.append(bi_gram)
            parents.append(group_name)
            values.append(count * subtheme_amplification_factor)

    # Create Sunburst Chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hoverinfo="label+value+percent entry",
    ))

    # Customize layout
    fig.update_layout(
        margin=dict(t=25, l=10, r=10, b=10),
        title="Radial Sunburst of Grouped Topics and Sub-Themes",
        width=1000,
        height=1000
    )

    # Save and show the chart
    fig.write_html("grouped_radial_sunburst_chart_bi_gram.html")

# Call the function
generate_grouped_sunburst_chart(topic_model, document_info)


# # Sunburst Chart Function
# def generate_sunburst_chart(topic_model, selected_topics, document_info):
#     # Prepare the data for the sunburst
#     labels = []
#     parents = []
#     values = []

#     for topic_id in selected_topics:
#         # Add main topic
#         topic_name = f"Topic {topic_id}"
#         labels.append(topic_name)
#         parents.append("")
#         topic_value = len(document_info[document_info["Topic"] == topic_id]) * 0.1  # Number of documents
#         values.append(topic_value)

#         # Add sub-themes (bi-grams) for each topic
#         sub_themes = topic_model.get_topic(topic_id)  # Extract sub-themes
#         if sub_themes:
#             for sub_theme, weight in sub_themes[:10]:  # Limit sub-themes to the top 10
#                 labels.append(sub_theme)
#                 parents.append(topic_name)
#                 values.append(weight)  # Use weight as value for sub-themes

#     # Create Sunburst Chart
#     fig = go.Figure(go.Sunburst(
#         labels=labels,
#         parents=parents,
#         values=values,
#         branchvalues="total",
#         hoverinfo="label+value+percent entry",
#     ))

#     # Customize layout
#     fig.update_layout(
#         margin=dict(t=10, l=10, r=10, b=10),
#         title="Radial Sunburst of Topics and Sub-Themes"
#     )

#     # Save and show the chart
#     fig.write_html("radial_sunburst_chart.html")

# # Extract the final content from selected documents
# selected_docs = selected_documents["Document"].tolist()

# # Generate top bi-grams for the selected documents
# bi_grams = generate_bi_grams(selected_docs, top_n=50)

# # Call the sunburst chart function
# generate_sunburst_chart(topic_model, selected_topics, document_info)




