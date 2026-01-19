# =====================================================================
# PART 1: CREATE PROFESSIONAL VECTOR DATABASE (FAISS + PARQUET)
#          -- LOCAL FILE STORAGE --
# =====================================================================

# 1. Ensure necessary libraries are installed via your terminal:
#    pip install sentence-transformers pandas numpy tqdm faiss-cpu pyarrow

# 2. Imports
import pandas as pd
import numpy as np
import os
import io
import faiss  # The industry standard for vector search
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# --- Local File Setup ---
# IMPORTANT: Replace 'rag_table_csv_cleaned.csv' with the actual path
# to your local CSV file.
LOCAL_INPUT_FILE = "rag_table_csv_cleaned.csv"

# IMPORTANT: Change this to your desired local output directory.
LOCAL_OUTPUT_DIR = "../Data/football_FAISS/"
print(f"✓ Database will be saved to: {os.path.abspath(LOCAL_OUTPUT_DIR)}")

# ---------------------- 1. Load and Prepare Data ---------------------- #
print(f"\nLoading local CSV file: {LOCAL_INPUT_FILE}")
try:
    rag_table = pd.read_csv(LOCAL_INPUT_FILE)
    print(f"Loaded data with shape: {rag_table.shape}")
except FileNotFoundError:
    print(f"Error: Input file not found at {LOCAL_INPUT_FILE}")
    print("Please ensure the path to your CSV is correct.")
    raise
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise

rag_columns = [
    # Identity (NEW: Added 'Comp' for League/Competition)
    "Player", "Club", "Nation", "Pos", "Season", "Age", "League",

    # Playing Time
    "Playing_Time_MP", "Playing_Time_90s",

    # Performance
    "Performance_Gls", "Performance_Ast", "Performance_G+A",

    # Per 90 Stats
    "Per_90_Minutes_Gls", "Per_90_Minutes_Ast", "Per_90_Minutes_G+A",

    # Expected Stats
    "Expected_xG", "Expected_npxG", "Expected_xAG", "Expected_npxG+xAG",

    # Shooting
    "Standard_Sh", "Standard_SoT", "Standard_Sh/90", "Standard_SoT/90",

    # Passing
    "Total_Cmp%", "Total_TotDist", "Total_PrgDist",

    # Creativity
    "KP", "xAG", "Expected_xA", "PPA",

    # Progression
    "Progression_PrgC", "Progression_PrgP", "Progression_PrgR",

    # Shot/Goal Creation
    "SCA_SCA", "SCA_SCA90", "GCA", "GCA90"
]

# Filter available columns
available_cols = [col for col in rag_columns if col in rag_table.columns]
rag_data = rag_table[available_cols].copy()

# Ensure 'Comp' exists in the document, even if the user's data didn't have it
if 'Comp' not in rag_data.columns:
    rag_data['Comp'] = rag_data.get('League', 'N/A') # Use League if Comp not present
print(f"Filtered data shape: {rag_data.shape}. Note: 'Comp' assumed for League data.")


# ---------------------- 2. Create Documents ---------------------- #
def create_player_document(row) -> str:
    def safe_val(val): return val if pd.notna(val) else "N/A"

    doc = f"""Football Player Analysis: {safe_val(row.get('Player'))}
Identity: Name: {safe_val(row.get('Player'))}, Club: {safe_val(row.get('Club'))}, League: {safe_val(row.get('League'))}, Nation: {safe_val(row.get('Nation'))}, Position: {safe_val(row.get('Pos'))}, Season: {safe_val(row.get('Season'))}
Key Performance Metrics:
- Matches: {safe_val(row.get('Playing_Time_MP'))}
- Goals: {safe_val(row.get('Performance_Gls'))} (xG: {safe_val(row.get('Expected_xG'))})
- Assists: {safe_val(row.get('Performance_Ast'))} (xAG: {safe_val(row.get('Expected_xAG'))})
- Shooting: {safe_val(row.get('Standard_Sh'))} shots
- Passing: {safe_val(row.get('Total_Cmp%'))}% completion
- Creativity: {safe_val(row.get('KP'))} Key Passes
- Progression: {safe_val(row.get('Progression_PrgC'))} Carries
"""
    return doc.strip()


print("\n📝 Creating text documents...")
documents = [create_player_document(row) for _, row in rag_data.iterrows()]
metadata = rag_data.to_dict('records')

print(f"Created {len(documents)} documents")

print("PyTorch detected GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))


# ---------------------- 3. Generate Embeddings ---------------------- #
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
print(f"\nLoading Superior Embedding Model: {MODEL_NAME}")
# Automatically uses GPU if available
model = SentenceTransformer(MODEL_NAME)

print("\nGenerating high-quality embeddings...")
# Add the device parameter to explicitly use CUDA if available, or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=device
)

faiss.normalize_L2(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# ---------------------- 4. Build FAISS Index ---------------------- #
print("\nBuilding FAISS Index...")
# We use IndexFlatIP for Inner Product (Cosine Similarity on L2-normalized vectors)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"FAISS Index built with {index.ntotal} vectors")


# ---------------------- 5. Save Database Locally ---------------------- #
print("\nSaving Vector Database Locally...")

# Create the directory if it doesn't exist
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# A. Save the FAISS Index (The Vectors)
index_path_local = os.path.join(LOCAL_OUTPUT_DIR, "football.index")
faiss.write_index(index, index_path_local)

# B. Save the Data (Text + Metadata) as Parquet
df_save = pd.DataFrame(metadata)
df_save['text_document'] = documents
data_path_local = os.path.join(LOCAL_OUTPUT_DIR, "football_data.parquet")
df_save.to_parquet(data_path_local)


print("\nVECTOR DATABASE CREATION COMPLETE!")
print("="*80)
print(f"Files successfully saved to: {os.path.abspath(LOCAL_OUTPUT_DIR)}")
print(f"  1. FAISS Index: {os.path.basename(index_path_local)}")
print(f"  2. Data/Metadata: {os.path.basename(data_path_local)}")
print("="*80)