import streamlit as st # For UI feedback from tool functions - try to minimize/refactor later
from langchain.tools import Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel # For llm type hint

# Attempt to import from the new structure
try:
    from src.utils.retrieval_utils import expand_query, rerank_documents
    from src.db_managers.vector_store_manager import semantic_search_chroma, search_supabase_store
    from src.config.app_config import RETRIEVAL_INITIAL_TOP_K, TAVILY_MAX_RESULTS, ARXIV_DEFAULT_MAX_DOCS
    # RERANKING_TOP_N_SELECT and QUERY_EXPANSION_NUM_QUERIES are used by functions in retrieval_utils directly
except ImportError: # Fallback for local testing if src is not in PYTHONPATH
    from utils.retrieval_utils import expand_query, rerank_documents
    from db_managers.vector_store_manager import semantic_search_chroma, search_supabase_store
    from src.config.app_config import (
        RETRIEVAL_INITIAL_TOP_K,
        TAVILY_MAX_RESULTS,
        ARXIV_DEFAULT_MAX_DOCS,
        TAVILY_API_KEY # Import TAVILY_API_KEY
    )
    # RERANKING_TOP_N_SELECT and QUERY_EXPANSION_NUM_QUERIES are used by functions in retrieval_utils directly
except ImportError: # Fallback for local testing if src is not in PYTHONPATH
    # This block is for when running tool_definitions.py directly and src.config is not in path
    print("WARN: tool_definitions.py running with fallback imports/configs for direct testing.")
    from utils.retrieval_utils import expand_query, rerank_documents
    from db_managers.vector_store_manager import semantic_search_chroma, search_supabase_store
    RETRIEVAL_INITIAL_TOP_K = 5
    TAVILY_MAX_RESULTS = 3
    ARXIV_DEFAULT_MAX_DOCS = 3
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Need os import for this fallback
    if TAVILY_API_KEY is None: import os # Ensure os is imported if TAVILY_API_KEY is None then used


# Initialize Arxiv Tool globally within this module as it's stateless
# In app.py, this was: arxiv_tool = ArxivQueryRun()
# We'll make it a function to be called in get_all_tools or keep it global if preferred.
# For now, let's assume it's simple enough to be initialized directly in get_all_tools.

# PDF RAG Tool Function (Session ChromaDB)
def query_uploaded_pdfs_func(original_query: str, llm: BaseChatModel, pdf_collection: object, st_objects: dict) -> str:
    """
    Tool function to query PDFs uploaded in the current session (ChromaDB).
    Needs access to the LLM for expansion/reranking and synthesis,
    the session's PDF collection (ChromaDB), and Streamlit objects for feedback.
    """
    if pdf_collection is None:
        return "No PDF documents have been uploaded for the current session. Please upload PDFs using the sidebar."

    # Use st_objects for Streamlit calls to allow testing without st context
    st_info = st_objects.get('info', print)
    st_write = st_objects.get('write', print)
    st_spinner = st_objects.get('spinner', lambda x: type('dummy_spinner', (object,), {'__enter__': lambda: None, '__exit__': lambda *a: None})())


    st_info(f"Enhancing query for session PDF search: '{original_query}'")
    expanded_queries = expand_query(original_query, llm, num_expansions=2)
    st_write(f"Expanded queries for session PDFs: {expanded_queries}")

    all_retrieved_doc_texts = []
    retrieved_doc_ids = set()

    with st_spinner(f"Searching session PDFs with expanded queries for: '{original_query}'..."):
        for i, exp_query in enumerate(expanded_queries):
            st_write(f"Searching session PDFs with: \"{exp_query}\" (Expansion {i+1}/{len(expanded_queries)})")
            # semantic_search_chroma will need the actual SentenceTransformer model, not the LC wrapper
            # This needs to be passed or made accessible. For now, assuming it's handled by semantic_search_chroma's setup
            # This highlights a dependency: semantic_search_chroma needs the raw embedding model.
            # Let's assume it's passed via st_objects or semantic_search_chroma can get it.
            # For now, this will likely break unless semantic_search_chroma is adapted or model passed.
            # HACK: For now, this function won't work until embedding model for chroma is plumbed.
            # This should be: results = semantic_search_chroma(exp_query, pdf_collection, st_objects['embedding_model_st'], top_k=RETRIEVAL_INITIAL_TOP_K)
            results = {} # Placeholder
            if 'embedding_model_st' in st_objects and pdf_collection:
                 results = semantic_search_chroma(exp_query, pdf_collection, st_objects['embedding_model_st'], top_k=RETRIEVAL_INITIAL_TOP_K)
            else:
                print("ERROR: embedding_model_st not found in st_objects for query_uploaded_pdfs_func")
                return "Error: Session PDF search is not properly configured (missing embedding model)."


            if results and results.get('documents') and results['documents'][0]:
                current_query_docs = results['documents'][0]
                current_query_ids = results['ids'][0]
                for doc_id, doc_text in zip(current_query_ids, current_query_docs):
                    if doc_id not in retrieved_doc_ids:
                        all_retrieved_doc_texts.append(doc_text)
                        retrieved_doc_ids.add(doc_id)

    if not all_retrieved_doc_texts:
        return f"No relevant information found in the currently uploaded PDF documents for: '{original_query}' (after query expansion)."

    temp_lc_documents = []
    for i, content_str in enumerate(all_retrieved_doc_texts):
        metadata = {"source": "chroma_session_pdf", "retrieved_id": list(retrieved_doc_ids)[i] if i < len(retrieved_doc_ids) else f"text_match_{i}"}
        temp_lc_documents.append(Document(page_content=content_str, metadata=metadata))

    st_info(f"Re-ranking {len(temp_lc_documents)} retrieved session PDF documents...")
    reranked_lc_documents = rerank_documents(original_query, temp_lc_documents, llm, top_n_to_select=3)

    if not reranked_lc_documents:
        return f"Could not determine the most relevant documents from session PDFs for '{original_query}' after re-ranking."

    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_lc_documents])

    st_info("Synthesizing answer from re-ranked session PDF context...")
    prompt_text = f"Based ONLY on the following highly relevant context from uploaded PDF documents:\n\nContext:\n{context}\n\nAnswer the following query: {original_query}"
    try:
        response = llm.invoke(prompt_text)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        # st_error = st_objects.get('error', print)
        # st_error(f"LLM error generating response from re-ranked session PDF context: {e}")
        print(f"LLM error in query_uploaded_pdfs_func: {e}")
        return f"Error generating response from re-ranked session PDF context: {str(e)}"

# ArXiv Search Tool
def search_arxiv_papers_func(query: str, arxiv_tool_instance: ArxivQueryRun) -> str:
    """ Wrapper for ArxivQueryRun to be used as a tool function. """
    try:
        return arxiv_tool_instance.invoke(query)
    except Exception as e:
        print(f"Error in search_arxiv_papers_func: {e}")
        return f"Error searching ArXiv: {str(e)}"

# Long-Term Memory (Supabase) Tool
def query_long_term_memory_func(original_query: str, llm: BaseChatModel, vector_store: object, st_objects: dict) -> str:
    """
    Tool function to query Supabase long-term memory.
    Needs access to LLM, Supabase vector store, and Streamlit objects for feedback.
    """
    if vector_store is None:
        return "Long-term memory (Supabase) is not available or configured."

    st_info = st_objects.get('info', print)
    st_write = st_objects.get('write', print)
    st_spinner = st_objects.get('spinner', lambda x: type('dummy_spinner', (object,), {'__enter__': lambda: None, '__exit__': lambda *a: None})())

    st_info(f"Enhancing query for long-term memory search: '{original_query}'")
    expanded_queries = expand_query(original_query, llm, num_expansions=2)
    st_write(f"Expanded queries for LTM: {expanded_queries}")

    all_retrieved_docs = []
    retrieved_doc_content_hashes = set()

    with st_spinner(f"Searching long-term memory with expanded queries for: '{original_query}'..."):
        for i, exp_query in enumerate(expanded_queries):
            st_write(f"Searching LTM with: \"{exp_query}\" (Expansion {i+1}/{len(expanded_queries)})")
            retrieved_docs_for_exp_query = search_supabase_store(vector_store, exp_query, top_k=RETRIEVAL_INITIAL_TOP_K)

            for doc in retrieved_docs_for_exp_query:
                content_hash = hash(doc.page_content)
                if content_hash not in retrieved_doc_content_hashes:
                    all_retrieved_docs.append(doc)
                    retrieved_doc_content_hashes.add(content_hash)

    if not all_retrieved_docs:
        return f"No relevant information found in long-term memory for: '{original_query}' (after query expansion)."

    st_info(f"Re-ranking {len(all_retrieved_docs)} retrieved long-term memory documents...")
    reranked_ltm_documents = rerank_documents(original_query, all_retrieved_docs, llm, top_n_to_select=3)

    if not reranked_ltm_documents:
         return f"Could not determine the most relevant documents from long-term memory for '{original_query}' after re-ranking."

    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_ltm_documents])

    st_info("Synthesizing answer from re-ranked long-term memory context...")
    prompt_text = f"Based ONLY on the following highly relevant context from the long-term knowledge base:\n\nContext:\n{context}\n\nAnswer the following query: {original_query}"
    try:
        response = llm.invoke(prompt_text)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        # st_error = st_objects.get('error', print)
        # st_error(f"LLM error generating response from re-ranked long-term memory context: {e}")
        print(f"LLM error in query_long_term_memory_func: {e}")
        return "Error generating response from re-ranked long-term memory context."

# Function to get all tools for the agent
def get_all_tools(
    llm: BaseChatModel,
    pdf_session_collection: object | None, # ChromaDB Collection
    supabase_vector_store: object | None, # SupabaseVectorStore instance
    # tavily_api_key is now imported from config
    st_embedding_model: object | None # Direct SentenceTransformer model for Chroma
) -> list:
    """
    Initializes and returns a list of all tools available to the agent.
    Uses TAVILY_API_KEY from app_config.
    """
    tools = []

    # Wrapper for Streamlit objects to pass to tool functions if they need UI feedback
    # This helps in decoupling tool logic from direct Streamlit calls, making them more testable.
    # In a more advanced setup, logging or a dedicated feedback mechanism might be used.
    streamlit_feedback_objects = {
        'info': st.info if 'st' in globals() else print,
        'write': st.write if 'st' in globals() else print,
        'spinner': st.spinner if 'st' in globals() else lambda x: type('dummy_spinner', (object,), {'__enter__': lambda: None, '__exit__': lambda *a: None})(),
        'error': st.error if 'st' in globals() else print,
        'embedding_model_st': st_embedding_model # Crucial for session PDF tool
    }

    # PDF Session RAG Tool
    tools.append(Tool(
        name="QueryUploadedPDFs",
        func=lambda query_str: query_uploaded_pdfs_func(query_str, llm, pdf_session_collection, streamlit_feedback_objects),
        description="Use this tool to answer questions based on the content of PDF documents that the user has uploaded *during the current session*. If no PDFs are uploaded, inform the user to upload them first."
    ))

    # ArXiv Search Tool
    # Pass ARXIV_DEFAULT_MAX_DOCS if ArxivQueryRun accepts it, or handle truncation if needed.
    # ArxivQueryRun default is load_max_docs=3. We can customize if the tool allows.
    # For now, assuming ArxivQueryRun uses its own defaults or we can wrap it if specific control needed.
    # Let's check ArxivQueryRun documentation. It has `max_docs_returned` in constructor.
    # However, the instance `arxiv_tool` was initialized globally in old app.py.
    # For now, we'll use a new instance here if we want to pass args.
    # Or, if `arxiv_tool` is imported from `llm_utils` or similar, it would be configured there.
    # Let's assume we initialize it here for clarity of parameter usage.
    arxiv_tool_instance = ArxivQueryRun(load_max_docs=ARXIV_DEFAULT_MAX_DOCS)
    tools.append(Tool(
        name="SearchArXiv",
        func=lambda query_str: search_arxiv_papers_func(query_str, arxiv_tool_instance),
        description=f"Use this tool to search for academic papers on ArXiv. Returns up to {ARXIV_DEFAULT_MAX_DOCS} paper summaries."
    ))

    # Tavily Web Search Tool
    if TAVILY_API_KEY: # Use imported constant
        tavily_search_tool = TavilySearchResults(
            api_key=TAVILY_API_KEY,
            max_results=TAVILY_MAX_RESULTS,
            name="WebSearch"
        )
        tavily_search_tool.description = f"A search engine optimized for comprehensive, accurate, and trusted results. Use this for general web searches and up-to-date information. Returns top {TAVILY_MAX_RESULTS} results."
        tools.append(tavily_search_tool)
    else:
        # UI warning about Tavily key missing is handled in streamlit_app.py's sidebar
        print("INFO: TAVILY_API_KEY not found (via app_config). Web search tool will not be added.")


    # Long-Term Memory (Supabase) Tool
    if supabase_vector_store:
        tools.append(Tool(
            name="QueryLongTermMemory",
            func=lambda query_str: query_long_term_memory_func(query_str, llm, supabase_vector_store, streamlit_feedback_objects),
            description="Searches and retrieves information from the persistent long-term knowledge base (Supabase). Use this for queries about previously processed documents or general knowledge accumulated over time. Not for current session PDFs unless they have been explicitly stored here."
        ))
        print("INFO: Long-term memory tool (QueryLongTermMemory) added to agent.")
    # else:
    #     # UI warning about Supabase missing is handled in streamlit_app.py's sidebar
    #     print("INFO: Long-term memory tool (QueryLongTermMemory) not added as Supabase vector store is not available.")

    return tools

if __name__ == '__main__':
    print("Testing tool_definitions.py...")
    # This file is not meant to be run directly without a proper context
    # (LLM, vector stores, Streamlit session state etc.)
    # Basic check:
    # if callable(get_all_tools):
    #    print("get_all_tools function is defined.")
    #    # Dummy objects for testing the structure of get_all_tools
    #    class DummyLLM(BaseChatModel):
    #        def _generate(self, messages, stop=None, run_manager=None, **kwargs): pass
    #        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs): pass
    #        @property
    #        def _llm_type(self) -> str: return "dummy"

    #    class DummyCollection: pass
    #    class DummyVectorStore: pass
    #    class DummySTModel: pass

    #    dummy_llm = DummyLLM()
    #    dummy_pdf_collection = DummyCollection()
    #    dummy_supabase_store = DummyVectorStore()
    #    dummy_st_model = DummySTModel()

    #    all_tools = get_all_tools(dummy_llm, dummy_pdf_collection, dummy_supabase_store, "dummy_tavily_key", dummy_st_model)
    #    print(f"Successfully called get_all_tools. Number of tools returned: {len(all_tools)}")
    #    for tool in all_tools:
    #        print(f"Tool: {tool.name}, Description: {tool.description[:60]}...")
    print("tool_definitions.py test finished (conceptual).")
