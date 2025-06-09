# src/config/app_config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load .env file at the top

# --- Environment Variables (loaded from .env file) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") # Retained as it was in .env.example

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Cloudflare R2 Configuration
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# Langfuse Observability Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000") # Default for local development


# --- Application Behavior Constants ---
# LLM and Embedding Models
GEMINI_LLM_MODEL_NAME = "gemini-1.5-flash"
# Default temperature for the LLM, can be overridden at initialization if needed
LLM_DEFAULT_TEMPERATURE = 0.1
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Used by SentenceTransformer

# RAG Retrieval Parameters
# For initial vector search (semantic_search_chroma, search_supabase_store)
RETRIEVAL_INITIAL_TOP_K = 5 # Number of documents to fetch initially per query/expansion
# For query expansion
QUERY_EXPANSION_NUM_QUERIES = 2  # Number of *alternative* queries to generate (total queries will be this + 1 original)
# For re-ranking step
RERANKING_TOP_N_SELECT = 3       # Number of documents to select after re-ranking to pass to final LLM

# Vector Store Configurations
SUPABASE_DEFAULT_TABLE_NAME = "documents"
SUPABASE_QUERY_NAME = "match_documents" # Default function name for SupabaseVectorStore
CHROMA_SESSION_COLLECTION_NAME = "pdf_session_storage" # Default for session ChromaDB (matches streamlit_app.py)
# Path for ChromaDB persistent client (used in session_manager.py)
CHROMA_PERSISTENT_PATH = "chroma_db_refactored"


# Text Chunking Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Agent Configuration
AGENT_MAX_ITERATIONS = 5

# R2 Object Storage
R2_UPLOAD_FOLDER = "uploads" # Subfolder within the R2 bucket

# Add any other constants that were previously hardcoded and make sense to centralize.
# For example, default model names or parameters used across multiple modules.
# Default ArXiv search parameters (if any were hardcoded)
ARXIV_DEFAULT_MAX_DOCS = 3 # Example if ArxivQueryRun had a max_docs parameter being used

# Default Tavily search parameters
TAVILY_MAX_RESULTS = 3

# Summarization Trigger
# Max characters in retrieved tool context before summarization is triggered
TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION = 5000
# Max characters for the input to the summarizer graph itself
SUMMARIZER_MAX_INPUT_CHARS = 10000

# Agent Behavior
MAX_TOOL_RETRIES = 2  # Max number of retries for a failing tool (total attempts = 1 + MAX_TOOL_RETRIES)

# Logging Configuration
LOG_LEVEL = "INFO"  # Default log level (e.g., DEBUG, INFO, WARNING, ERROR)


# Print a message if run directly, to confirm it's accessible
if __name__ == '__main__':
    print("app_config.py loaded. Constants are:")
    print(f"  GEMINI_LLM_MODEL_NAME: {GEMINI_LLM_MODEL_NAME}")
    print(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"  RETRIEVAL_INITIAL_TOP_K: {RETRIEVAL_INITIAL_TOP_K}")
    print(f"  QUERY_EXPANSION_NUM_QUERIES: {QUERY_EXPANSION_NUM_QUERIES}")
    print(f"  RERANKING_TOP_N_SELECT: {RERANKING_TOP_N_SELECT}")
    print(f"  CHROMA_SESSION_COLLECTION_NAME: {CHROMA_SESSION_COLLECTION_NAME}")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION: {TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION}")
    print(f"  SUMMARIZER_MAX_INPUT_CHARS: {SUMMARIZER_MAX_INPUT_CHARS}")
    print(f"  LANGFUSE_PUBLIC_KEY: {'*' * 5 if LANGFUSE_PUBLIC_KEY else 'Not set'}") # Avoid printing actual keys
    print(f"  LANGFUSE_SECRET_KEY: {'*' * 5 if LANGFUSE_SECRET_KEY else 'Not set'}")
    print(f"  LANGFUSE_HOST: {LANGFUSE_HOST}")
