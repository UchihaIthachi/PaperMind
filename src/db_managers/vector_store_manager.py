import os
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.embeddings import Embeddings # Using base class for type hint
from langchain_core.documents import Document
# from dotenv import load_dotenv # No longer needed here, app_config handles it
import streamlit as st # For UI feedback - marked for later refactoring
import chromadb # For session store
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for ChromaDB text splitting
from sentence_transformers import SentenceTransformer # Added for ChromaDB embeddings
from src.config.app_config import (
    SUPABASE_URL, SUPABASE_ANON_KEY, # Import env var constants
    SUPABASE_DEFAULT_TABLE_NAME,
    SUPABASE_QUERY_NAME,
    CHROMA_SESSION_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS
)

# load_dotenv() # Handled by app_config.py

# --- Supabase Client Initialization ---
def get_supabase_client() -> Client | None:
    """
    Connects to Supabase using environment variables (via app_config).
    Returns a Supabase client instance or None if connection fails or vars are missing.
    """
    # Use imported constants
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("WARNING: SUPABASE_URL or SUPABASE_ANON_KEY not found (loaded by app_config). Supabase client cannot be initialized.")
        return None
    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        # print("INFO: Supabase client successfully created.") # Can be noisy, enable if needed
        return client
    except Exception as e:
        print(f"ERROR: Error connecting to Supabase: {e}")
        return None

# --- Supabase Vector Store Initialization ---
def get_supabase_vector_store(
    embedding_model: Embeddings,
    table_name: str = SUPABASE_DEFAULT_TABLE_NAME,
    client: Client | None = None,
    query_name: str = SUPABASE_QUERY_NAME
) -> SupabaseVectorStore | None:
    """
    Initializes and returns a SupabaseVectorStore instance using app_config defaults.
    (SQL setup instructions as previously defined are assumed to be in this docstring)
    --- IMPORTANT SQL SETUP ---
    Users must run the following SQL in their Supabase SQL editor BEFORE using this.
    Replace 'public.documents' with your desired table name if different.
    Replace 'vector(384)' with the correct dimension for your embedding model (e.g., 'all-MiniLM-L6-v2' is 384).

    1. Enable the vector extension:
    CREATE EXTENSION IF NOT EXISTS vector;

    2. Create the documents table:
    CREATE TABLE IF NOT EXISTS public.documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content TEXT,
        metadata JSONB,
        embedding VECTOR(384) -- Ensure this matches your embedding model's dimension
    );

    3. Create the match_documents function:
    CREATE OR REPLACE FUNCTION match_documents (
        query_embedding VECTOR(384), -- Match this to your embedding model's dimension
        match_count INT,
        filter JSONB DEFAULT '{}'::jsonb
    )
    RETURNS TABLE (
        id UUID,
        content TEXT,
        metadata JSONB,
        similarity FLOAT
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT
            documents.id,
            documents.content,
            documents.metadata,
            1 - (documents.embedding <=> query_embedding) AS similarity
        FROM
            documents
        WHERE
            (metadata @> filter OR filter = '{}'::jsonb)
        ORDER BY
            documents.embedding <=> query_embedding
        LIMIT match_count;
    END;
    $$;
    --- END SQL SETUP ---
    """
    if client is None:
        client = get_supabase_client()

    if client is None:
        print("WARNING: Supabase client is not available. Cannot initialize SupabaseVectorStore.")
        return None

    if embedding_model is None:
        print("WARNING: Embedding model is not provided. Cannot initialize SupabaseVectorStore.")
        return None

    try:
        vector_store = SupabaseVectorStore(
            client=client,
            embedding=embedding_model,
            table_name=table_name,
            query_name=query_name,
        )
        print(f"INFO: Successfully initialized SupabaseVectorStore for table '{table_name}'.")
        return vector_store
    except Exception as e:
        print(f"ERROR: Error initializing SupabaseVectorStore: {e}")
        print("Please ensure you have run the required SQL setup in your Supabase project (see comments in get_vector_store).")
        return None

# --- Supabase Document Storage Functions ---
def add_texts_to_supabase_store(
    vector_store: SupabaseVectorStore | None,
    texts: list[str],
    metadatas: list[dict] | None = None
):
    if not vector_store:
        print("WARNING: Supabase vector store not available. Skipping document addition.")
        st.error("Long-term memory store (Supabase) not available. Cannot add documents.") # UI Feedback
        return
    if not texts:
        print("INFO: No texts provided to add to Supabase store.")
        return
    try:
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        print(f"INFO: Successfully added {len(texts)} text chunks to Supabase.")
        st.success(f"Successfully added {len(texts)} text chunks to long-term memory (Supabase)!") # UI Feedback
    except Exception as e:
        print(f"ERROR: Error adding documents to SupabaseVectorStore: {e}")
        st.error(f"Error adding documents to long-term memory (Supabase): {e}") # UI Feedback

def search_supabase_store(
    vector_store: SupabaseVectorStore | None,
    query: str,
    top_k: int = 3
) -> list[Document]:
    if not vector_store:
        print("WARNING: Supabase vector store not available. Skipping search.")
        return []
    try:
        results = vector_store.similarity_search(query, k=top_k)
        print(f"INFO: Found {len(results)} documents in Supabase for query: '{query}'")
        return results
    except Exception as e:
        print(f"ERROR: Error searching SupabaseVectorStore: {e}")
        return []

# --- ChromaDB Session Store Functions ---

def process_and_store_chunks_in_chroma(
    all_text: str,
    chroma_client: chromadb.API,
    embedding_model_st: SentenceTransformer,
    collection_name: str = CHROMA_SESSION_COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_separators: list[str] | None = None # Allow None to use default in splitter
) -> tuple[list[str], chromadb.Collection | None]:
    """
    Splits text, stores embeddings in a session-specific ChromaDB collection, and returns the text chunks.
    Uses app_config defaults for collection name and chunking parameters.
    """
    if not all_text:
        return [], None
    if not chroma_client or not embedding_model_st:
        print("ERROR: ChromaDB client or SentenceTransformer embedding model not provided to process_and_store_chunks_in_chroma.")
        st.error("Session store components not ready. Cannot process PDFs for session.") # UI Feedback
        return [], None

    actual_separators = chunk_separators if chunk_separators is not None else CHUNK_SEPARATORS

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=actual_separators,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    if not chunks:
        # st.warning("Text extraction yielded no processable chunks for ChromaDB.") # UI Feedback
        print("INFO: Text extraction yielded no processable chunks for ChromaDB.")
        return [], None

    # st.info(f"Split text into {len(chunks)} chunks for session RAG (ChromaDB).") # UI Feedback
    print(f"INFO: Split text into {len(chunks)} chunks for ChromaDB collection '{collection_name}'.")

    try:
        valid_collection_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in collection_name)
        if len(valid_collection_name) < 3 or len(valid_collection_name) > 63:
             valid_collection_name = f"session_coll_{hash(collection_name) % 10000}"

        try: # Try to get collection first, if it exists, delete it for a fresh start this session
            existing_collection = chroma_client.get_collection(name=valid_collection_name)
            if existing_collection:
                chroma_client.delete_collection(name=valid_collection_name)
                print(f"INFO: Deleted existing ChromaDB collection: {valid_collection_name}")
        except Exception: # Collection might not exist, which is fine
            pass

        collection = chroma_client.create_collection(
            name=valid_collection_name,
            # embedding_function=None # Not needed if providing embeddings directly
        )

        embeddings = embedding_model_st.encode(chunks).tolist()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                ids=[f"session_chunk_{valid_collection_name}_{i}"], # More unique ID
                embeddings=[embedding],
                metadatas=[{"source": "pdf_session_upload", "chunk_id": i, "collection": valid_collection_name}],
                documents=[chunk]
            )
        # The collection object is stored in st.session_state by the calling code in app_logic or streamlit_app.py
        # e.g., st.session_state[collection_name] = collection
        print(f"INFO: Stored {len(chunks)} chunks in session ChromaDB collection: {valid_collection_name}")
        return chunks, collection
    except Exception as e:
        # st.error(f"Error processing text for session ChromaDB: {e}") # UI Feedback
        print(f"ERROR: Error processing text for ChromaDB session store: {e}")
        return chunks, None # Return chunks even if DB fails, so Supabase can still try

def semantic_search_chroma(
    query: str,
    collection: chromadb.Collection | None,
    embedding_model_st: SentenceTransformer, # Expects a SentenceTransformer model instance
    top_k: int = 2
) -> dict:
    """
    Performs semantic search on a ChromaDB collection.
    """
    if not collection:
        print("WARNING: ChromaDB collection not provided for semantic_search_chroma.")
        return {} # Return empty dict for consistency with ChromaDB's possible empty result
    if not embedding_model_st:
        print("ERROR: SentenceTransformer embedding model not provided for semantic_search_chroma.")
        # st.error("Embedding model not available for session search.") # UI Feedback
        return {}

    try:
        query_embedding = embedding_model_st.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        # print(f"INFO: ChromaDB semantic search returned {len(results.get('ids', [[]])[0])} results.")
        return results
    except Exception as e:
        print(f"ERROR: Error during ChromaDB semantic search: {e}")
        return {}


if __name__ == '__main__':
    print("Testing vector_store_manager.py...")
    # (Supabase test code as before, potentially add ChromaDB specific tests here too if needed)
    print("vector_store_manager.py test finished.")
