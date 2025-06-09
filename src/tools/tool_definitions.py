import streamlit as st # For UI feedback from tool functions - try to minimize/refactor later
from langchain.tools import Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel # For llm type hint

from langfuse import Langfuse # Added for custom span creation
import json # For serializing complex objects for metadata

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
def query_uploaded_pdfs_func(original_query: str, llm: BaseChatModel, st_objects: dict) -> str:
    """
    Tool function to query PDFs uploaded in the current session (ChromaDB).
    Needs access to the LLM for expansion/reranking and synthesis,
    Streamlit objects for feedback, and st.session_state for the PDF collection.
    """
    session_state = st_objects.get('session_state')
    langfuse_client = None
    if session_state and hasattr(session_state, 'langfuse_client'):
        langfuse_client = session_state.langfuse_client

    def try_update_span(span, **kwargs):
        if span:
            try:
                span.update(**kwargs)
            except Exception as e:
                print(f"Langfuse: Error updating span: {e}")

    def try_end_span(span, **kwargs):
        if span:
            try:
                span.end(**kwargs)
            except Exception as e:
                print(f"Langfuse: Error ending span: {e}")

    pdf_collection = getattr(session_state, 'pdf_session_collection', None)
    if pdf_collection is None:
        return "No PDF documents have been uploaded for the current session. Please upload PDFs using the sidebar."

    # Query Expansion
    expanded_queries = []
    pdf_query_expansion_span = None
    try:
        if langfuse_client:
            pdf_query_expansion_span = langfuse_client.span(name="pdf_query_expansion", input={'query': original_query})
        expanded_queries = expand_query(original_query, llm, num_expansions=2)
        try_end_span(pdf_query_expansion_span, output={'expanded_queries': expanded_queries, 'count': len(expanded_queries)})
    except Exception as e:
        print(f"Error during PDF query expansion: {e}")
        try_end_span(pdf_query_expansion_span, level='ERROR', status_message=str(e), output={'expanded_queries': [], 'count': 0})
        # Decide if to proceed with original query or return error
        expanded_queries = [original_query] # Fallback to original query

    all_retrieved_doc_texts = []
    retrieved_doc_ids = set()
    initial_search_span = None
    try:
        if langfuse_client:
            initial_search_span = langfuse_client.span(name="initial_vector_search_chroma", input={'expanded_queries': expanded_queries})

        for i, exp_query in enumerate(expanded_queries):
            query_specific_span = None
            if langfuse_client:
                query_specific_span = initial_search_span.span(name=f"chroma_search_sub_query_{i+1}", input={'query': exp_query})

            results = {}
            if 'embedding_model_st' in st_objects and pdf_collection:
                results = semantic_search_chroma(exp_query, pdf_collection, st_objects['embedding_model_st'], top_k=RETRIEVAL_INITIAL_TOP_K)
            else:
                print("ERROR: embedding_model_st not found in st_objects for query_uploaded_pdfs_func")
                # This error should ideally be caught by the main try-except or handled differently
                # For now, if it occurs, it might bypass span ending.
                return "Error: Session PDF search is not properly configured (missing embedding model)."

            doc_texts_for_query = []
            if results and results.get('documents') and results['documents'][0]:
                current_query_docs = results['documents'][0]
                current_query_ids = results['ids'][0]
                for doc_id, doc_text in zip(current_query_ids, current_query_docs):
                    if doc_id not in retrieved_doc_ids:
                        all_retrieved_doc_texts.append(doc_text)
                        retrieved_doc_ids.add(doc_id)
                        doc_texts_for_query.append(doc_text[:100] + "..." if len(doc_text) > 100 else doc_text) # Log truncated

            try_end_span(query_specific_span, output={'retrieved_doc_count': len(doc_texts_for_query), 'retrieved_texts_preview': doc_texts_for_query})

        try_end_span(initial_search_span, output={'total_unique_docs_retrieved': len(all_retrieved_doc_texts)})
    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        try_end_span(initial_search_span, level='ERROR', status_message=str(e))
        # Potentially return error or empty results if search fails critically
        if not all_retrieved_doc_texts: # If search failed before retrieving anything
             return f"Error searching session PDFs: {str(e)}"


    if not all_retrieved_doc_texts:
        return f"No relevant information found in the currently uploaded PDF documents for: '{original_query}' (after query expansion)."

    temp_lc_documents = [Document(page_content=text, metadata={"id": doc_id}) for doc_id, text in zip(list(retrieved_doc_ids), all_retrieved_doc_texts)]

    # Reranking
    reranked_lc_documents = []
    pdf_reranking_span = None
    try:
        if langfuse_client:
            pdf_reranking_span = langfuse_client.span(name="pdf_reranking", input={'doc_count': len(temp_lc_documents), 'query': original_query})
        reranked_lc_documents = rerank_documents(original_query, temp_lc_documents, llm, top_n_to_select=3)
        try_end_span(pdf_reranking_span, output={'reranked_doc_count': len(reranked_lc_documents),
                                                 'reranked_docs_preview': [d.page_content[:100]+"..." for d in reranked_lc_documents]})
    except Exception as e:
        print(f"Error during PDF reranking: {e}")
        try_end_span(pdf_reranking_span, level='ERROR', status_message=str(e))
        # Fallback: use non-reranked documents if reranking fails? Or return error.
        # For now, if reranking fails, it might proceed with empty reranked_lc_documents. This needs careful thought.
        if not reranked_lc_documents: # If reranking failed and returned nothing
            return f"Error reranking PDF documents: {str(e)}"


    if not reranked_lc_documents:
        return f"Could not determine the most relevant documents from session PDFs for '{original_query}' after re-ranking."

    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_lc_documents])
    prompt_text = f"Based ONLY on the following highly relevant context from uploaded PDF documents:\n\nContext:\n{context}\n\nAnswer the following query: {original_query}"

    # Final LLM Synthesis
    final_llm_span = None
    try:
        if langfuse_client:
            final_llm_span = langfuse_client.span(name="pdf_synthesis_llm_call", input={'prompt_length': len(prompt_text), 'context_docs_count': len(reranked_lc_documents)})
        response = llm.invoke(prompt_text)
        result_content = response.content if hasattr(response, 'content') else str(response)
        try_end_span(final_llm_span, output={'response_length': len(result_content), 'response_preview': result_content[:100]+"..."})
        return result_content
    except Exception as e:
        print(f"LLM error in query_uploaded_pdfs_func synthesis: {e}")
        try_end_span(final_llm_span, level='ERROR', status_message=str(e))
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
    session_state = st_objects.get('session_state')
    langfuse_client = None
    if session_state and hasattr(session_state, 'langfuse_client'):
        langfuse_client = session_state.langfuse_client

    def try_update_span(span, **kwargs):
        if span:
            try:
                span.update(**kwargs)
            except Exception as e:
                print(f"Langfuse: Error updating span: {e}")

    def try_end_span(span, **kwargs):
        if span:
            try:
                span.end(**kwargs)
            except Exception as e:
                print(f"Langfuse: Error ending span: {e}")

    if vector_store is None:
        return "Long-term memory (Supabase) is not available or configured."

    # Query Expansion
    expanded_queries = []
    ltm_query_expansion_span = None
    try:
        if langfuse_client:
            ltm_query_expansion_span = langfuse_client.span(name="ltm_query_expansion", input={'query': original_query})
        expanded_queries = expand_query(original_query, llm, num_expansions=2)
        try_end_span(ltm_query_expansion_span, output={'expanded_queries': expanded_queries, 'count': len(expanded_queries)})
    except Exception as e:
        print(f"Error during LTM query expansion: {e}")
        try_end_span(ltm_query_expansion_span, level='ERROR', status_message=str(e), output={'expanded_queries': [], 'count': 0})
        expanded_queries = [original_query] # Fallback

    all_retrieved_docs = [] # Stores Langchain Document objects
    retrieved_doc_content_hashes = set()
    initial_search_span_ltm = None
    try:
        if langfuse_client:
            initial_search_span_ltm = langfuse_client.span(name="initial_vector_search_supabase", input={'expanded_queries': expanded_queries})

        for i, exp_query in enumerate(expanded_queries):
            query_specific_span_ltm = None
            if langfuse_client:
                query_specific_span_ltm = initial_search_span_ltm.span(name=f"supabase_search_sub_query_{i+1}", input={'query': exp_query})

            retrieved_docs_for_exp_query = search_supabase_store(vector_store, exp_query, top_k=RETRIEVAL_INITIAL_TOP_K)

            docs_preview_for_query = []
            for doc in retrieved_docs_for_exp_query:
                content_hash = hash(doc.page_content) # Simple way to check for uniqueness based on content
                if content_hash not in retrieved_doc_content_hashes:
                    all_retrieved_docs.append(doc) # Store the full Document object
                    retrieved_doc_content_hashes.add(content_hash)
                    docs_preview_for_query.append(doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content)

            try_end_span(query_specific_span_ltm, output={'retrieved_doc_count': len(docs_preview_for_query),
                                                          'retrieved_docs_preview': docs_preview_for_query})
        try_end_span(initial_search_span_ltm, output={'total_unique_docs_retrieved': len(all_retrieved_docs)})
    except Exception as e:
        print(f"Error during Supabase search: {e}")
        try_end_span(initial_search_span_ltm, level='ERROR', status_message=str(e))
        if not all_retrieved_docs:
            return f"Error searching long-term memory: {str(e)}"


    if not all_retrieved_docs:
        return f"No relevant information found in long-term memory for: '{original_query}' (after query expansion)."

    # Reranking
    reranked_ltm_documents = []
    ltm_reranking_span = None
    try:
        if langfuse_client:
            ltm_reranking_span = langfuse_client.span(name="ltm_reranking", input={'doc_count': len(all_retrieved_docs), 'query': original_query})
        # Pass the actual Document objects to rerank_documents
        reranked_ltm_documents = rerank_documents(original_query, all_retrieved_docs, llm, top_n_to_select=3)
        try_end_span(ltm_reranking_span, output={'reranked_doc_count': len(reranked_ltm_documents),
                                                 'reranked_docs_preview': [d.page_content[:100]+"..." for d in reranked_ltm_documents]})
    except Exception as e:
        print(f"Error during LTM reranking: {e}")
        try_end_span(ltm_reranking_span, level='ERROR', status_message=str(e))
        if not reranked_ltm_documents:
             return f"Error reranking LTM documents: {str(e)}"

    if not reranked_ltm_documents:
         return f"Could not determine the most relevant documents from long-term memory for '{original_query}' after re-ranking."

    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_ltm_documents])
    prompt_text = f"Based ONLY on the following highly relevant context from the long-term knowledge base:\n\nContext:\n{context}\n\nAnswer the following query: {original_query}"

    # Final LLM Synthesis
    final_llm_span_ltm = None
    try:
        if langfuse_client:
            final_llm_span_ltm = langfuse_client.span(name="ltm_synthesis_llm_call", input={'prompt_length': len(prompt_text), 'context_docs_count': len(reranked_ltm_documents)})
        response = llm.invoke(prompt_text)
        result_content = response.content if hasattr(response, 'content') else str(response)
        try_end_span(final_llm_span_ltm, output={'response_length': len(result_content), 'response_preview': result_content[:100]+"..."})
        return result_content
    except Exception as e:
        print(f"LLM error in query_long_term_memory_func synthesis: {e}")
        try_end_span(final_llm_span_ltm, level='ERROR', status_message=str(e))
        return f"Error generating response from re-ranked long-term memory context: {str(e)}"

# Function to get all tools for the agent
def get_all_tools(
    llm: BaseChatModel,
    # pdf_session_collection is no longer directly passed; it's accessed via st.session_state
    supabase_vector_store: object | None, # SupabaseVectorStore instance
    # tavily_api_key is now imported from config
    st_embedding_model: object | None # Direct SentenceTransformer model for Chroma
) -> list:
    """
    Initializes and returns a list of all tools available to the agent.
    Uses TAVILY_API_KEY from app_config.
    The session PDF collection (ChromaDB) is accessed dynamically from st.session_state.
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
        'embedding_model_st': st_embedding_model, # Crucial for session PDF tool
        'session_state': st.session_state if 'st' in globals() else None # Provide access to session_state
    }

    # PDF Session RAG Tool
    tools.append(Tool(
        name="QueryUploadedPDFs",
        func=lambda query_str: query_uploaded_pdfs_func(query_str, llm, streamlit_feedback_objects),
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

    #    # Adjusted call for testing as pdf_session_collection is removed from direct args
    #    all_tools = get_all_tools(dummy_llm, dummy_supabase_store, dummy_st_model)
    #    print(f"Successfully called get_all_tools. Number of tools returned: {len(all_tools)}")
    #    for tool in all_tools:
    #        print(f"Tool: {tool.name}, Description: {tool.description[:60]}...")
    print("tool_definitions.py test finished (conceptual).")
