import streamlit as st
import os # For checking env vars for R2 status

# Attempt to import from the new structure
try:
    from src.utils.llm_utils import get_llm, get_embedding_model
    from src.db_managers.vector_store_manager import get_supabase_client, get_supabase_vector_store
    from src.db_managers.file_object_store import get_r2_client
    from src.config.app_config import CHROMA_PERSISTENT_PATH # Added import
except ImportError: # Fallback for local testing if src is not in PYTHONPATH
    # This might happen if running session_manager.py directly for tests without `src` in path
    print("WARN: session_manager running with fallback imports. Ensure 'src' is in PYTHONPATH for app execution.")
    from utils.llm_utils import get_llm, get_embedding_model
    from db_managers.vector_store_manager import get_supabase_client, get_supabase_vector_store
    from db_managers.file_object_store import get_r2_client
    # Assuming app_config might also need a fallback if this script is run standalone
    # However, direct execution is for testing, and config might be found differently or mocked.
    # For simplicity, only handling the primary import path for CHROMA_PERSISTENT_PATH here.
    # If direct testing of this script needs it, it would require specific test setup.
    try:
        from config.app_config import CHROMA_PERSISTENT_PATH
    except ImportError:
        # Fallback if run from a context where src.config is not directly available
        # This is a simple attempt, real testing might need a proper path setup or mocking
        print("WARN: Could not import CHROMA_PERSISTENT_PATH from config.app_config in fallback.")
        CHROMA_PERSISTENT_PATH = "chroma_db_refactored" # Default fallback path



def initialize_session_state():
    """
    Initializes all necessary keys in Streamlit's session state.
    This includes LLM, embedding models, DB clients, vector stores, and chat memory.
    """

    # Initialize LLM
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm() # Handles its own API key check

    # Initialize Embedding Models
    if "embedding_model_st" not in st.session_state or "embedding_model_lc" not in st.session_state:
        st_model, lc_model = get_embedding_model()
        st.session_state.embedding_model_st = st_model # Direct SentenceTransformer
        st.session_state.embedding_model_lc = lc_model # Langchain wrapper

    # Initialize Supabase Client and Vector Store
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = get_supabase_client()

    if "supabase_vector_store" not in st.session_state:
        if st.session_state.supabase_client and st.session_state.embedding_model_lc:
            st.session_state.supabase_vector_store = get_supabase_vector_store(
                client=st.session_state.supabase_client,
                embedding_model=st.session_state.embedding_model_lc
                # table_name and query_name use defaults from get_supabase_vector_store
            )
            if st.session_state.supabase_vector_store:
                print("INFO: Supabase vector store initialized and stored in session state.")
            else:
                print("WARN: Supabase vector store initialization failed but client was available.")
        else:
            st.session_state.supabase_vector_store = None
            print("INFO: Supabase client or Langchain embedding model not available, Supabase vector store not initialized.")

    # Initialize R2 Client
    if "r2_client" not in st.session_state:
        st.session_state.r2_client = get_r2_client()
        if st.session_state.r2_client:
            print("INFO: R2 client initialized and stored in session state.")
        else:
            print("INFO: R2 client not initialized (check .env config).")


    # Initialize Session-specific PDF RAG collection (ChromaDB)
    # The actual ChromaDB client is initialized globally in vector_store_manager for now,
    # or should be passed around. For session collection, it's often managed per session.
    # Let's ensure it's None initially; app logic will create it on PDF upload.
    if "pdf_session_collection" not in st.session_state:
        st.session_state.pdf_session_collection = None
        # This will be populated by process_and_store_chunks_in_chroma via streamlit_app.py

    # Initialize ChromaDB client (PersistentClient for session stores)
    # This was globally initialized in the old app.py.
    # It's better to have it managed, perhaps here or passed to where it's needed.
    # For now, let's assume vector_store_manager might initialize its own or accept one.
    # The current process_and_store_chunks_in_chroma expects a client.
    # This implies the app should manage one instance of ChromaDB client.
    if "chromadb_client" not in st.session_state:
        try:
            # Reusing the path from the old app.py for the persistent client
            # This client will be passed to functions in vector_store_manager that need it.
            import chromadb
            st.session_state.chromadb_client = chromadb.PersistentClient(path=CHROMA_PERSISTENT_PATH) # Use constant
            print("INFO: ChromaDB persistent client initialized for session stores.")
        except Exception as e:
            print(f"ERROR: Failed to initialize ChromaDB persistent client: {e}")
            st.session_state.chromadb_client = None


if __name__ == '__main__':
    print("Testing session_manager.py...")
    # This test needs to be run in a Streamlit context to access st.session_state.
    # You would typically call initialize_session_state() from your main Streamlit app.

    # Conceptual test (cannot run st.session_state outside Streamlit):
    # print("Attempting to initialize session state (conceptual)...")
    # initialize_session_state()
    # print("Session state keys might include (if successful):")
    # print("- llm")
    # print("- embedding_model_st, embedding_model_lc")
    # print("- supabase_client, supabase_vector_store")
    # print("- r2_client")
    # print("- pdf_session_collection (initially None)")
    # print("- chromadb_client")
    print("Run this as part of a Streamlit app to test st.session_state interactions.")
    print("\nsession_manager.py test finished.")
