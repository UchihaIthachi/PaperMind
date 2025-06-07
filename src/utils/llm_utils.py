from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as LangchainSentenceTransformerEmbeddings # To distinguish from direct SentenceTransformer
from sentence_transformers import SentenceTransformer # For direct use, e.g. with ChromaDB if not using LC wrapper

# Environment variables are loaded by app_config.py
from src.config.app_config import (
    GEMINI_API_KEY, # Import the environment variable constant
    GEMINI_LLM_MODEL_NAME,
    LLM_DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL_NAME
)

_llm_instance = None
_embedding_model_instance_lc = None # Langchain wrapper
_embedding_model_instance_st = None # Direct SentenceTransformer

def get_llm(model_name: str = GEMINI_LLM_MODEL_NAME,
            temperature: float = LLM_DEFAULT_TEMPERATURE) -> ChatGoogleGenerativeAI | None:
    """
    Initializes and returns a ChatGoogleGenerativeAI LLM instance.
    Caches the instance for efficiency.
    Uses defaults from app_config.
    """
    global _llm_instance
    if _llm_instance: # and matches current config if config could change? For now, simple cache.
        # If model_name or temperature could change dynamically per call and require re-init,
        # this caching logic would need to be more sophisticated.
        # For now, assume they are fixed after first call or that a single instance is fine.
        return _llm_instance

    # Use the imported GEMINI_API_KEY constant
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found (loaded by app_config). LLM cannot be initialized.")
        # In a Streamlit context, you might use st.error() here, but this is a util module.
        # Raising an error or returning None are options. For now, returning None.
        return None

    try:
        _llm_instance = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY, # Use the constant
            temperature=temperature,
            # convert_system_message_to_human=True # May be needed for some older models/versions
        )
        print(f"INFO: ChatGoogleGenerativeAI LLM initialized (model: {model_name}).")
        return _llm_instance
    except Exception as e:
        print(f"ERROR: Failed to initialize ChatGoogleGenerativeAI LLM (model: {model_name}): {e}")
        return None

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> tuple[SentenceTransformer | None, Embeddings | None]:
    """
    Initializes and returns both a direct SentenceTransformer model and its Langchain wrapper.
    Caches instances for efficiency. Uses default from app_config.

    Args:
        model_name: The name of the SentenceTransformer model.

    Returns:
        A tuple containing:
            - SentenceTransformer instance (for direct use like local ChromaDB embeddings).
            - Langchain Embeddings wrapper instance (for use with Langchain components like SupabaseVectorStore).
        Returns (None, None) if initialization fails.
    """
    global _embedding_model_instance_st, _embedding_model_instance_lc

    if _embedding_model_instance_st and _embedding_model_instance_lc:
        return _embedding_model_instance_st, _embedding_model_instance_lc

    try:
        # Initialize direct SentenceTransformer model
        if _embedding_model_instance_st is None:
            _embedding_model_instance_st = SentenceTransformer(model_name)
            print(f"INFO: Direct SentenceTransformer model initialized ({model_name}).")

        # Initialize Langchain wrapper for the same model
        if _embedding_model_instance_lc is None:
            _embedding_model_instance_lc = LangchainSentenceTransformerEmbeddings(model_name=model_name)
            print(f"INFO: Langchain SentenceTransformerEmbeddings wrapper initialized ({model_name}).")

        return _embedding_model_instance_st, _embedding_model_instance_lc
    except Exception as e:
        print(f"ERROR: Failed to initialize embedding model '{model_name}': {e}")
        # Potentially add st.error here if called from Streamlit context and immediate feedback is desired.
        return None, None

if __name__ == '__main__':
    print("Testing llm_utils.py...")

    # Test LLM initialization (requires GEMINI_API_KEY)
    llm_instance = get_llm()
    if llm_instance:
        print("LLM instance retrieved successfully.")
        # You could try a simple invoke if needed, but ensure it doesn't consume too many resources
        # try:
        #     response = llm_instance.invoke("Hello!")
        #     print(f"LLM smoke test response: {response.content[:50]}...")
        # except Exception as e:
        #     print(f"LLM smoke test failed: {e}")
    else:
        print("LLM instance is None. Check GEMINI_API_KEY and logs.")

    # Test Embedding model initialization
    st_model, lc_model = get_embedding_model()
    if st_model and lc_model:
        print("Both SentenceTransformer and Langchain embedding models retrieved successfully.")
        # Test embedding a simple query with both
        # query = "Test embedding query"
        # try:
        #     st_embedding = st_model.encode(query)
        #     print(f"ST Model embedding dimension: {len(st_embedding)}")
        #     lc_embedding = lc_model.embed_query(query)
        #     print(f"LC Model embedding dimension: {len(lc_embedding)}")
        # except Exception as e:
        #     print(f"Embedding model test failed: {e}")
    else:
        print("One or both embedding models are None. Check logs.")

    # Test caching (call again)
    print("\nTesting caching by calling get_llm() and get_embedding_model() again...")
    llm_instance_cached = get_llm()
    st_model_cached, lc_model_cached = get_embedding_model()

    if llm_instance_cached is llm_instance:
        print("LLM instance caching confirmed.")
    else:
        print("LLM instance caching FAILED or first instance was None.")

    if st_model_cached is st_model and lc_model_cached is lc_model:
        print("Embedding model instances caching confirmed.")
    else:
        print("Embedding model instances caching FAILED or first instances were None.")

    print("\nllm_utils.py test finished.")
