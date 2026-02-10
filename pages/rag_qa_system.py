# =====================================================================
# PAGE 1: RAG Q&A SYSTEM
# Football Analytics RAG with FAISS + Hugging Face
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import List, Dict

# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="⚽ RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)

# =====================================================================
# Lazy Imports (Only load when this page is accessed)
# =====================================================================
@st.cache_resource
def lazy_import_rag_dependencies():
    """Import RAG-specific dependencies only when needed."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from dotenv import load_dotenv
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Load environment variables
        load_dotenv()
        
        return {
            'faiss': faiss,
            'SentenceTransformer': SentenceTransformer,
            'ChatHuggingFace': ChatHuggingFace,
            'HuggingFaceEndpoint': HuggingFaceEndpoint,
            'ChatPromptTemplate': ChatPromptTemplate,
            'StrOutputParser': StrOutputParser,
            'env_loaded': True
        }
    except ImportError as e:
        st.error(f"""
        ❌ RAG dependencies not found: {e}
        
        Please install required packages:
        ```
        pip install faiss-cpu sentence-transformers python-dotenv langchain-huggingface
        ```
        """)
        return None

# =====================================================================
# Configuration
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    HF_MODEL_REPO_ID = "deepseek-ai/DeepSeek-V3.2"
    HF_API_TOKEN = os.environ.get("HUGGING_FACE_API_TOKEN")
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    
    DB_PATH = os.path.join(BASE_DIR, "Data", "football_FAISS")
    DATA_FILE = os.path.join(DB_PATH, "football_data.parquet")
    INDEX_FILE = os.path.join(DB_PATH, "football.index")

# =====================================================================
# Initialization Functions
# =====================================================================
@st.cache_resource
def load_database():
    """Loads FAISS index, documents, and embedding model."""
    deps = lazy_import_rag_dependencies()
    if deps is None:
        st.stop()
    
    faiss = deps['faiss']
    SentenceTransformer = deps['SentenceTransformer']
    
    try:
        original_df = pd.read_parquet(Config.DATA_FILE)
        documents = original_df['text_document'].tolist()
        index = faiss.read_index(Config.INDEX_FILE)
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        
        st.success(f"✅ Database Loaded: {len(original_df)} records, Index with {index.ntotal} vectors.")
        return original_df, documents, index, embedding_model, deps
    
    except FileNotFoundError:
        st.error(f"❌ Database files not found at {Config.DB_PATH}. Please check paths.")
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Unexpected error during database loading: {e}")
        st.stop()

@st.cache_resource
def initialize_hf_chain(repo_id, api_token, _deps):
    """Initializes the LangChain RAG chain using Hugging Face Inference Endpoint."""
    
    if not api_token:
        st.error("❌ HUGGING_FACE_API_TOKEN is not set in the environment. Cannot initialize LLM.")
        return None
    
    ChatHuggingFace = _deps['ChatHuggingFace']
    HuggingFaceEndpoint = _deps['HuggingFaceEndpoint']
    
    try:
        hf_llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=api_token,
            temperature=0.3,
        )
        
        hf_chat_model = ChatHuggingFace(llm=hf_llm)
        st.success(f"✅ Hugging Face Chat Chain initialized with {repo_id}")
        
        return hf_chat_model
    
    except Exception as e:
        st.warning(f"⚠️ Hugging Face Initialization Failed: {e}")
        return None

# =====================================================================
# RAG Core Functions
# =====================================================================
def retrieve_players(query: str, k: int, filtered_df: pd.DataFrame, 
                    original_df: pd.DataFrame, index, embedding_model, 
                    documents: List[str], faiss):
    """Finds relevant players using FAISS on the filtered DataFrame."""
    
    df_to_search = filtered_df
    if len(df_to_search) == 0:
        return []
    
    # 1. Vectorize Query
    query_vector = embedding_model.encode([query])
    faiss.normalize_L2(query_vector)
    
    # 2. Prepare Filtered Vector Subset
    original_indices_to_search = df_to_search.index.values
    
    try:
        filtered_vectors = np.array([index.reconstruct(int(i)) for i in original_indices_to_search])
    except Exception as e:
        st.error(f"❌ Error during vector reconstruction: {e}")
        return []
    
    D = filtered_vectors.shape[1]
    temp_index = faiss.IndexFlatIP(D)
    temp_index.add(filtered_vectors)
    
    # 3. Search the temporary index
    distances, temp_indices = temp_index.search(query_vector, min(k, temp_index.ntotal))
    
    # 4. Map relative indices back to original indices
    results = []
    for i, temp_idx in enumerate(temp_indices[0]):
        if temp_idx != -1 and temp_idx < len(original_indices_to_search):
            original_idx = original_indices_to_search[temp_idx]
            player_data = original_df.iloc[original_idx]
            
            results.append({
                "rank": i + 1,
                "player": player_data['Player'],
                "club": player_data['Club'],
                "season": player_data['Season'],
                "text": documents[original_idx],
                "score": float(distances[0][i])
            })
    
    return results

def generate_answer(query: str, retrieved_docs: List[Dict], hf_chat_model, _deps):
    """Generates natural language answer using Hugging Face LLM."""
    
    if not hf_chat_model:
        return "❌ Error: Hugging Face generation chain not active. Check API token and model."
    
    ChatPromptTemplate = _deps['ChatPromptTemplate']
    StrOutputParser = _deps['StrOutputParser']
    
    context_text = "\n\n".join(
        [f"[Player {r['rank']} - {r['player']} ({r['club']}, {r['season']})]: {r['text']}" 
         for r in retrieved_docs]
    )
    
    try:
        RAG_SYSTEM_PROMPT = """
You are a professional football data analyst working inside a Retrieval-Augmented Generation (RAG) system.

You must answer the user's question using ONLY the information explicitly present in the provided Context Stats and Player Data.

STRICT RULES:
1. Do NOT use outside knowledge, assumptions, or real-world facts that are not present in the context.
2. If required information is missing, incomplete, or not present, respond exactly with:
   "I don't have that data".
3. You MAY ignore or discard context information that is irrelevant to the user's question.
4. You MAY use your reasoning abilities ONLY to:
    - compare players present in the context
    - filter out unrelated stats
    - summarize or rank based on available numbers
5. Do NOT invent statistics, seasons, players, or values.

STAT USAGE:
    - Cite numerical statistics (xG, Goals, Assists, Key Passes, Progressive Passes, etc.) ONLY when the user's question explicitly asks for:
      comparisons, rankings, justification, or numerical analysis.
    - Do NOT cite numbers for opinion-based or descriptive questions unless explicitly required.

OUTPUT FORMAT:
    - Provide ONLY the final answer.
    - Do NOT repeat the question.
    - Do NOT add headings, bullet points, markdown, or formatting tags.
    - Do NOT mention the words "context", "documents", or "data source".
    - Keep the response concise, precise, and analytical.

FAILURE CONDITION:
    If the answer cannot be confidently derived from the provided information, respond exactly with:
    "I don't have that data".
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Context Stats:\n{context}\n\nQuestion: {question}"),
        ])
        
        hf_chain = prompt | hf_chat_model | StrOutputParser()
        
        with st.spinner("🧠 AI Thinking... Generating Analysis..."):
            response = hf_chain.invoke({"context": context_text, "question": query})
        
        return response.strip()
    
    except Exception as e:
        return f"❌ Error invoking Hugging Face chain: {e}"

# =====================================================================
# Main App
# =====================================================================
def main():
    st.title("🤖 Football Analytics RAG Q&A System")
    st.markdown(f"""
    **Architecture:** Metadata Filtering → Vector Search (FAISS) → Generation (Hugging Face/{Config.HF_MODEL_REPO_ID})
    """)
    
    # Load Resources
    original_df, documents, index, embedding_model, deps = load_database()
    hf_chat_model = initialize_hf_chain(Config.HF_MODEL_REPO_ID, Config.HF_API_TOKEN, deps)
    use_llm = hf_chat_model is not None
    
    # Sidebar Filters
    st.sidebar.header("🎯 Metadata Filters")
    st.sidebar.info("Filters narrow the dataset before vector search. Use 'ALL' to ignore.")
    
    unique_seasons = sorted(original_df['Season'].unique().tolist(), reverse=True)
    unique_clubs = sorted(original_df['Club'].unique().tolist())
    
    player_name_input = st.sidebar.text_input("Player Name (Partial Match)", "").strip()
    
    season_input = st.sidebar.selectbox(
        "Season",
        ['ALL'] + unique_seasons,
        index=0
    )
    
    club_input = st.sidebar.selectbox(
        "Club/Competition",
        ['ALL'] + unique_clubs,
        index=0
    )
    
    k_input = st.sidebar.slider(
        "Number of Context Documents (k)", 
        1, 10, 4, 
        help="The number of top-scoring documents retrieved from FAISS."
    )
    
    st.sidebar.markdown("---")
    
    # Query Input
    st.subheader("❓ Ask your Football Analytics Question")
    user_query = st.text_area(
        "Enter your semantic query:",
        height=100,
        placeholder="e.g., 'Who had the highest xG in the 2024 season?', 'Compare Pedri and Gavi'"
    )
    
    if st.button("🔍 Run RAG Analysis", type="primary") and user_query:
        
        # Apply Metadata Filtering
        current_df = original_df.copy()
        
        if player_name_input:
            current_df = current_df[
                current_df['Player'].str.contains(player_name_input, case=False, na=False)
            ]
        
        if season_input != 'ALL':
            try:
                target_season = int(season_input)
                current_df = current_df[current_df['Season'] == target_season]
            except ValueError:
                pass
        
        if club_input != 'ALL':
            current_df = current_df[
                current_df['Club'].str.contains(club_input, case=False, na=False)
            ]
        
        st.info(f"📊 Metadata Filter applied. **{len(current_df)}** records remaining for vector search (out of {len(original_df)} total).")
        
        if len(current_df) == 0:
            st.error("❌ No players match the specified metadata filters. Please adjust the sidebar filters.")
            return
        
        # Retrieval (Vector Search)
        with st.spinner(f"🔍 Performing vector search on the filtered set of {len(current_df)} documents..."):
            retrieved_results = retrieve_players(
                user_query, k_input, current_df, original_df, 
                index, embedding_model, documents, deps['faiss']
            )
        
        if not retrieved_results:
            st.error("❌ Vector search returned no relevant documents. Try broadening your query or filters.")
            return
        
        # Generation & Display
        st.markdown("---")
        col1, col2 = st.columns([3, 2])
        
        # AI Response
        with col1:
            st.subheader("🧠 AI Analytical Response")
            
            if use_llm:
                answer = generate_answer(user_query, retrieved_results, hf_chat_model, deps)
                st.markdown(answer)
            else:
                st.warning("⚠️ LLM generation is disabled. Showing retrieval context only.")
        
        # Retrieved Context
        with col2:
            st.subheader(f"📊 Retrieved Context ({len(retrieved_results)} Docs)")
            
            for r in retrieved_results:
                with st.expander(f"**Rank {r['rank']}** - {r['player']} ({r['club']}, {r['season']}) | Score: {r['score']:.4f}"):
                    st.text(r['text'])

if __name__ == '__main__':
    main()