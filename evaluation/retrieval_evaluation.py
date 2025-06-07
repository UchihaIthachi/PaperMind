import os
import sys
import numpy as np # For cosine similarity calculation (if implementing manually)
import matplotlib.pyplot as plt # For plotting results later
from sklearn.metrics.pairwise import cosine_similarity # For similarity calculation
from sentence_transformers import SentenceTransformer # For direct embedding generation

# Add src directory to Python path to allow importing from src
# This assumes the script is run from the project root or 'evaluation' directory.
# If run from 'evaluation', '..' goes to project root, then 'src'.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.llm_utils import get_llm # get_embedding_model from llm_utils returns LC wrapper
from src.utils.retrieval_utils import expand_query, rerank_documents
from src.config import app_config # To use configured model names, etc.
from langchain_core.documents import Document

# --- 1. Dataset Definition ---
# (Query, List of ground truth relevant snippet indices from SAMPLE_CORPUS)
EVALUATION_QUERIES = [
    {
        "query": "What are the main benefits of using LangGraph for AI agents?",
        "ground_truth_ids": [0, 1] # Indices from SAMPLE_CORPUS
    },
    {
        "query": "How can RAG systems be improved with query expansion?",
        "ground_truth_ids": [2, 3]
    },
    {
        "query": "Tell me about vector databases for semantic search.",
        "ground_truth_ids": [4, 5]
    },
    {
        "query": "What is the role of LLM in re-ranking documents?", # New query for reranking test
        "ground_truth_ids": [10] # Assuming a new document about re-ranking will be added
    }
]

SAMPLE_CORPUS_TEXT = [
    # LangGraph related (IDs 0, 1)
    "LangGraph allows building stateful, multi-actor applications with LLMs, offering cycles and better control flow.", # 0
    "Key advantages of LangGraph include its explicitness in defining agent steps and transitions, making complex agents easier to debug.", # 1
    # Query Expansion related (IDs 2, 3)
    "Query expansion techniques, like generating multiple query variations with an LLM, can improve retrieval recall in RAG systems.", # 2
    "By broadening search terms, query expansion helps uncover relevant documents that might be missed by the original query alone.", # 3
    # Vector DB related (IDs 4, 5)
    "Vector databases store data as high-dimensional vectors, enabling fast semantic similarity searches for applications like RAG.", # 4
    "These databases use embeddings to represent text, allowing for nuanced retrieval based on meaning rather than just keywords.", # 5
    # Distractor documents
    "Traditional SQL databases are excellent for structured data storage and querying.", # 6
    "Machine learning involves training models on data to make predictions or decisions.", # 7
    "The capital of France is Paris, a major European city known for its art and culture.", # 8
    "Photosynthesis is the process by which green plants use sunlight to synthesize foods.", # 9
    # Document for re-ranking query (ID 10)
    "LLM-based re-ranking evaluates initially retrieved documents against the original query to improve the relevance ordering of final results." # 10
]

# Convert text corpus to Langchain Document objects for consistency with reranker
SAMPLE_CORPUS_DOCUMENTS: List[Document] = [
    Document(page_content=text, metadata={"id": i, "source": "sample_corpus"})
    for i, text in enumerate(SAMPLE_CORPUS_TEXT)
]


# --- 2. Initialize Models (within a main execution block or function) ---
def initialize_models_for_evaluation():
    print("Initializing LLM and Embedding Model for evaluation...")
    llm = None
    embedding_model_st = None
    try:
        # Get LLM (used for query expansion and re-ranking)
        llm = get_llm() # Uses GEMINI_LLM_MODEL_NAME from app_config by default
        if llm is None:
            raise ValueError("LLM initialization failed. Check GEMINI_API_KEY.")

        # Initialize SentenceTransformer model directly for embedding generation in this script
        # get_embedding_model() from llm_utils returns (SentenceTransformer, LangchainWrapper)
        # We need the direct SentenceTransformer instance here.
        # Let's modify llm_utils.get_embedding_model to ensure it can be called
        # or just instantiate directly as planned.
        # For this script, direct instantiation is simpler if llm_utils is not modified.
        print(f"Loading SentenceTransformer: {app_config.EMBEDDING_MODEL_NAME}")
        embedding_model_st = SentenceTransformer(app_config.EMBEDDING_MODEL_NAME)

        print("Models for evaluation initialized successfully.")
        return llm, embedding_model_st
    except Exception as e:
        print(f"ERROR: Error initializing models for evaluation: {e}")
        print("This script requires GEMINI_API_KEY for LLM operations (re-ranking, expansion).")
        print("Ensure the embedding model can be downloaded (e.g., internet connection for SentenceTransformer).")
        sys.exit(1) # Exit if models can't be loaded

if __name__ == "__main__":
    print("--- Starting Retrieval Evaluation Script Setup ---")
    llm_eval, embedding_model_eval_st = initialize_models_for_evaluation()

    if llm_eval and embedding_model_eval_st:
        print("\nEvaluation script setup complete. LLM and embedding model loaded.")
        print(f"Number of evaluation queries: {len(EVALUATION_QUERIES)}")
        print(f"Size of sample corpus: {len(SAMPLE_CORPUS_DOCUMENTS)} documents")
        # (Further implementation for retrieval simulation and metric functions will go here)
    else:
        print("\nFailed to initialize models. Cannot proceed with evaluation script.")

    print("\n--- Retrieval Evaluation Script Setup Finished ---")


# --- 3. Retrieval Simulation Functions ---

def embed_corpus(corpus_docs: list[Document], embedding_model_st: SentenceTransformer) -> tuple[dict, dict]:
    """
    Pre-embeds all documents in the corpus.
    Input: list of Langchain Document objects, initialized SentenceTransformer model.
    Output: A tuple of (embeddings dictionary, doc_id_to_document_map dictionary).
    """
    print(f"\nEmbedding corpus of {len(corpus_docs)} documents...")
    embeddings = {}
    doc_id_map = {}
    for i, doc in enumerate(corpus_docs):
        doc_id = doc.metadata.get("id", i)
        if doc_id in embeddings:
            print(f"Warning: Duplicate doc_id {doc_id} found in corpus. Consider unique IDs if this is not intended.")
        embeddings[doc_id] = embedding_model_st.encode(doc.page_content)
        doc_id_map[doc_id] = doc
    print("Corpus embedding complete.")
    return embeddings, doc_id_map


def simulate_naive_vector_search(
    query: str,
    corpus_embeddings: dict,
    doc_id_map: dict,
    embedding_model_st: SentenceTransformer,
    top_k: int
) -> list[Document]:
    """
    Simulates a naive vector search against the pre-embedded corpus.
    """
    query_embedding = embedding_model_st.encode(query)

    similarities = {} # Store doc_id: similarity

    for doc_id, doc_embedding in corpus_embeddings.items():
        # Cosine similarity expects 2D arrays
        sim = cosine_similarity(query_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))[0][0]
        similarities[doc_id] = sim

    # Sort by similarity (descending)
    sorted_doc_ids = sorted(similarities, key=similarities.get, reverse=True)

    top_k_docs = []
    for doc_id in sorted_doc_ids[:top_k]:
        if doc_id in doc_id_map:
            top_k_docs.append(doc_id_map[doc_id])
        else:
            # This should not happen if corpus_embeddings and doc_id_map are from the same corpus
            print(f"Warning: doc_id {doc_id} from sorted similarities not found in doc_id_map.")

    return top_k_docs


def simulate_enhanced_retrieval(
    original_query: str,
    corpus_docs: list[Document], # Passed to naive search, and for final doc objects
    corpus_embeddings: dict, # Pre-computed embeddings
    doc_id_map: dict, # Map from doc_id to Document object
    llm, # Main LLM for expansion and re-ranking
    embedding_model_st: SentenceTransformer, # For embedding expanded queries
    top_k_per_expansion: int, # K for retrieval for each expanded query
    top_k_final_rerank: int # Number of docs to select after re-ranking
) -> list[Document]:
    """
    Simulates an enhanced retrieval pipeline including query expansion and LLM-based re-ranking.
    """
    print(f"\nRunning ENHANCED retrieval for query: '{original_query}'")
    # Use app_config for number of expansions, ensure it's imported or default here
    num_exp = app_config.QUERY_EXPANSION_NUM_QUERIES if hasattr(app_config, 'QUERY_EXPANSION_NUM_QUERIES') else 2
    expanded_queries = expand_query(original_query, llm, num_expansions=num_exp)
    print(f"  Expanded queries: {expanded_queries}")

    all_retrieved_docs_map = {} # Using dict to handle de-duplication by doc_id

    for eq_idx, exp_query in enumerate(expanded_queries):
        print(f"    Searching for expanded query ({eq_idx+1}/{len(expanded_queries)}): '{exp_query}'")
        retrieved_for_exp_query = simulate_naive_vector_search(
            exp_query, corpus_docs, corpus_embeddings, doc_id_map, embedding_model_st, top_k_per_expansion
        )
        for doc in retrieved_for_exp_query:
            # Use metadata ID if available, otherwise fallback to index (though embed_corpus should ensure IDs)
            doc_id = doc.metadata.get("id", corpus_docs.index(doc) if doc in corpus_docs else -1)
            if doc_id != -1 and doc_id not in all_retrieved_docs_map:
                all_retrieved_docs_map[doc_id] = doc
            elif doc_id == -1:
                 print(f"Warning: Document '{doc.page_content[:30]}...' could not be reliably ID'd for de-duplication.")

    unique_retrieved_docs = list(all_retrieved_docs_map.values())
    print(f"  Retrieved {len(unique_retrieved_docs)} unique documents after expansion.")

    if not unique_retrieved_docs:
        return []

    # Re-rank the unique retrieved documents
    print(f"  Re-ranking {len(unique_retrieved_docs)} documents with LLM...")
    # Use app_config for reranking top N, ensure it's imported or default here
    num_rerank = app_config.RERANKING_TOP_N_SELECT if hasattr(app_config, 'RERANKING_TOP_N_SELECT') else 3
    reranked_docs = rerank_documents(
        original_query, unique_retrieved_docs, llm, top_n_to_select=num_rerank
    )
    print(f"  Re-ranked to {len(reranked_docs)} documents.")
    return reranked_docs


# --- 4. Metric Calculation Functions ---

def calculate_precision_at_k(
    retrieved_docs: list[Document],
    ground_truth_doc_ids: list[any], # e.g., list of integers if IDs are integers
    k: int
) -> float:
    """
    Calculates Precision@K.

    Args:
        retrieved_docs: A list of Document objects retrieved by a search method.
                        Assumes doc.metadata['id'] exists and matches ground truth ID type.
        ground_truth_doc_ids: A list of document IDs considered relevant for the query.
        k: The number of top retrieved documents to consider.

    Returns:
        Precision at K (float).
    """
    if k == 0:
        return 0.0
    if not retrieved_docs:
        return 0.0

    top_k_retrieved_docs = retrieved_docs[:k]
    retrieved_ids_at_k = {doc.metadata.get("id") for doc in top_k_retrieved_docs if doc.metadata.get("id") is not None}

    if not retrieved_ids_at_k:
        return 0.0

    ground_truth_set = set(ground_truth_doc_ids)
    relevant_retrieved_count = 0
    for doc_id in retrieved_ids_at_k:
        if doc_id in ground_truth_set:
            relevant_retrieved_count += 1

    return relevant_retrieved_count / k


if __name__ == "__main__":
    print("--- Starting Retrieval Evaluation Script ---") # Updated title
    llm_eval, embedding_model_eval_st = initialize_models_for_evaluation()

    if llm_eval and embedding_model_eval_st:
        print("\nEvaluation script setup complete. LLM and embedding model loaded.")
        print(f"Number of evaluation queries: {len(EVALUATION_QUERIES)}")
        print(f"Size of sample corpus: {len(SAMPLE_CORPUS_DOCUMENTS)} documents")

        corpus_embeddings_map, doc_id_to_document_map = embed_corpus(SAMPLE_CORPUS_DOCUMENTS, embedding_model_eval_st)

        print("\n--- Running Full Evaluation Loop ---")

        # K value for Precision@K. This should align with how many results we ultimately care about.
        # For enhanced retrieval, this often matches top_k_final_rerank.
        eval_k_value = app_config.RERANKING_TOP_N_SELECT

        results_summary = []

        for i, query_info in enumerate(EVALUATION_QUERIES):
            query = query_info["query"]
            ground_truth_ids = query_info["ground_truth_ids"]
            print(f"\nProcessing Query {i+1}/{len(EVALUATION_QUERIES)}: '{query}'")
            print(f"  Ground Truth IDs: {ground_truth_ids}")

            # Naive Retrieval
            # For P@K, we should retrieve K items in the naive search as well for a fair comparison base for this metric
            naive_retrieved_docs = simulate_naive_vector_search(
                query,
                corpus_embeddings_map,
                doc_id_to_document_map,
                embedding_model_eval_st,
                top_k=eval_k_value # Retrieve K docs for P@K
            )
            precision_naive = calculate_precision_at_k(naive_retrieved_docs, ground_truth_ids, eval_k_value)
            print(f"  Naive P@{eval_k_value}: {precision_naive:.2f} (Retrieved {len(naive_retrieved_docs)} docs)")

            # Enhanced Retrieval
            # top_k_per_expansion should be >= eval_k_value to give reranker enough options.
            # Let's set it to eval_k_value + 1 or 2, e.g., 5 if eval_k_value is 3.
            # app_config.RETRIEVAL_INITIAL_TOP_K is 5.
            # app_config.RERANKING_TOP_N_SELECT is 3 (which is our eval_k_value).

            enhanced_retrieved_docs = simulate_enhanced_retrieval(
                query,
                SAMPLE_CORPUS_DOCUMENTS,
                corpus_embeddings_map,
                doc_id_to_document_map,
                llm_eval,
                embedding_model_eval_st,
                top_k_per_expansion=app_config.RETRIEVAL_INITIAL_TOP_K, # How many docs each expanded query retrieves
                top_k_final_rerank=eval_k_value  # How many docs the reranker should finally select, matching K for P@K
            )
            precision_enhanced = calculate_precision_at_k(enhanced_retrieved_docs, ground_truth_ids, eval_k_value)
            print(f"  Enhanced P@{eval_k_value}: {precision_enhanced:.2f} (Retrieved {len(enhanced_retrieved_docs)} docs after rerank)")

            results_summary.append({
                "query": query,
                "precision_naive": precision_naive,
                "precision_enhanced": precision_enhanced,
                "naive_docs_retrieved_count": len(naive_retrieved_docs),
                "enhanced_docs_retrieved_count": len(enhanced_retrieved_docs)
            })

        # Calculate Average Precision@K
        avg_precision_naive = np.mean([res["precision_naive"] for res in results_summary]) if results_summary else 0.0
        avg_precision_enhanced = np.mean([res["precision_enhanced"] for res in results_summary]) if results_summary else 0.0

        print("\n--- Evaluation Summary ---")
        print(f"K for Precision@K: {eval_k_value}")
        for res in results_summary:
            print(f"  Query: '{res['query']}'")
            print(f"    Naive P@{eval_k_value}: {res['precision_naive']:.2f} (Retrieved: {res['naive_docs_retrieved_count']})")
            print(f"    Enhanced P@{eval_k_value}: {res['precision_enhanced']:.2f} (Retrieved: {res['enhanced_docs_retrieved_count']})")

        print(f"\nAverage Naive P@{eval_k_value}: {avg_precision_naive:.2f}")
        print(f"Average Enhanced P@{eval_k_value}: {avg_precision_enhanced:.2f}")

        # Generate and save the plot
        plot_precision_comparison(avg_precision_naive, avg_precision_enhanced, eval_k_value)

        print("\nEvaluation and Precision@K visualization complete.") # Updated message
    else:
        print("\nFailed to initialize models. Cannot proceed with evaluation script.")

    print("\n--- Retrieval Evaluation Script Finished ---")


# --- 5. Visualization Function ---
def plot_precision_comparison(
    avg_p_naive: float,
    avg_p_enhanced: float,
    k_value: int,
    output_path: str = "evaluation/retrieval_precision_comparison.png" # Default output path
):
    """
    Generates and saves a bar chart comparing average Precision@K.
    """
    labels = ['Naive Retrieval', f'Enhanced Retrieval (w/ Re-ranking)']
    scores = [avg_p_naive, avg_p_enhanced]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6)) # Slightly larger figure
    rects = ax.bar(x, scores, width, label=f'Precision@{k_value}', color=['skyblue', 'lightcoral'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'Average Precision@{k_value}')
    ax.set_title(f'Average Precision@{k_value} Comparison for Retrieval Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(scores) * 1.15 if max(scores) > 0 else 1.0) # Dynamic y-limit
    ax.legend(loc='upper right')

    ax.bar_label(rects, padding=3, fmt='%.3f') # Add labels on top of bars, show 3 decimal places

    fig.tight_layout()

    # Ensure the output directory exists
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"INFO: Created directory: {output_dir}")

        plt.savefig(output_path)
        print(f"INFO: Comparison chart saved to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save plot to {output_path}: {e}")
    # plt.show() # Optionally show plot if running in an interactive environment
