import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

torch.backends.cudnn.benchmark = True

tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-2_R', load_in_8bit=True)
model.eval()  # Set model to evaluation mode

def print_memory_usage(description):
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"{description} - Memory Allocated: {allocated:.2f} GB; Memory Reserved: {reserved:.2f} GB")

print_memory_usage("Initial")

def generate_embeddings(texts):
    batch_dict = tokenizer(texts, max_length=1024, padding=True, truncation=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, -1, :]
        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

df = pd.read_csv("./climate_crawler/contrarian_cleaned_final.csv")
batch_size = 8  # Manageable batch size

print_memory_usage("Post Model Load")

# Prepare DataFrame for new column
df['embedding'] = None

# Handling output file
output_file = "final_embeddings_contrarian.csv"
with open(output_file, "w") as f:
    df.iloc[0:0].to_csv(f, index=False)  # Write headers only

    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        valid_rows = df.iloc[i:end]['clean_content'].dropna()
        batch_texts = valid_rows.loc[valid_rows.apply(lambda x: isinstance(x, str))].tolist()
        embeddings = generate_embeddings(batch_texts)

        # Ensure correct DataFrame updating
        for j, emb in enumerate(embeddings):
            df.at[i + j, 'embedding'] = list(emb)  # Assign each embedding individually

        # Write batch to file
        df.iloc[i:end].to_csv(f, mode='a', header=False, index=False)
        print_memory_usage(f"After Batch {(i // batch_size) + 1}")

        torch.cuda.empty_cache()

print("Embedding generation and saving to CSV completed.")
