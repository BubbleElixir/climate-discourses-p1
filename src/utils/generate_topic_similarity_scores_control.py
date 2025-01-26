import spacy
import pandas as pd

# Load SpaCy model
nlp = spacy.load('en_core_web_md')

keywords = ["climate change",  "global warming", "carbon emission", "greenhouse gas",  "carbon tax, “renewable energy", "carbon footprint", "fossil fuel", "climate crisis", "climate action", "extreme weather",  "clean energy", "carbon neutral", "zero emission", "net zero", "climate policy", "climate talk", "climate scientist",  "climate report",  "climate strategy",  "carbon tax",  "carbon capture", "climate risk", "climate emergency", "climate protest", "climate assembly", "climate jury", "climate citizen", "climate deliberation", "climate report", "climate recommendation", "climate policy", "climate science", "climate advocacy", "climate action", "sustainable policy", "environmental report", "climate coalition", "climate justice", "climate reparations", "climate education", "climate legislation", "environmental justice", "climate communication", "climate governance", "climate workshop", "climate strike", "climate dialogue", "climate protest", "climate rally", "climate myth", "climate fiction", "green agenda", "climate realism", "natural variability", "no warming", "climate propaganda",  "climate hysteria", "energy poverty"  "manufactured consensus", "climate scam", "plant food", "climate lies",  "climate religion" "economic burden", "temperature manipulation" "global cooling","alarmist agenda", "CO2 benefits","natural cycles" "climate model", "climate variability", "false consensus"]

def calculate_similarities(text):
    # Process the text with SpaCy to create a document object
    doc1 = nlp(text)
    # Dictionary to store scores
    similarity_scores = {}
    # Calculate similarity for each keyword
    for keyword in keywords:
        doc2 = nlp(keyword)
        # Calculate semantic similarity using SpaCy (both docs need vectors)
        similarity_scores[keyword] = doc1.similarity(doc2) if doc1.has_vector and doc2.has_vector else 0
    return similarity_scores

# Load your CSV file
df = pd.read_csv('activist_blogs_2.csv')

# Assuming 'content' is the column name with text data
df['similarity_scores'] = df['content'].apply(calculate_similarities)

# Save the updated dataframe to a new CSV file
df.to_csv('activist_blogs_topics.csv', index=False)