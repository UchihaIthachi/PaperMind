import os
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting the llm
from langchain_core.documents import Document # For type hinting
import re # For parsing LLM output

# Load environment variables if you were to use an LLM that needs API keys directly here
# from dotenv import load_dotenv
# load_dotenv()
from src.config.app_config import QUERY_EXPANSION_NUM_QUERIES, RERANKING_TOP_N_SELECT

def expand_query(original_query: str, llm: BaseChatModel, num_expansions: int = QUERY_EXPANSION_NUM_QUERIES) -> list[str]:
    """
    Expands a given query using an LLM to generate alternative phrasings or expansions.
    Uses app_config default for num_expansions.

    Args:
        original_query: The user's original query.
        llm: The language model instance to use for expansion.
        num_expansions: The desired number of alternative queries.

    Returns:
        A list of queries, including the original query and its expansions.
        Returns just the original query in a list if expansion fails or no expansions generated.
    """
    if not original_query or not llm:
        return [original_query] if original_query else []

    # Ensure num_expansions is positive
    if num_expansions <= 0:
        num_expansions = 1 # Default to at least one expansion if called with non-positive

    prompt_template = f"""Given the user query: '{original_query}'
Generate {num_expansions} alternative phrasings or expansions of this query that would be effective for searching a document database.
Focus on rephrasing, adding synonyms, or breaking down the query if it's complex.
Return each generated query on a new line. Do not include numbering, bullet points, or any other introductory text or formatting.
Only return the generated queries, each on a new line.

Example if user query is 'benefits of AI in healthcare' and num_expansions is 3:
AI applications in medical field
advantages of artificial intelligence in hospitals
how AI improves patient outcomes
"""

    try:
        # print(f"DEBUG: Query Expansion Prompt:\n{prompt_template}") # For debugging
        response = llm.invoke(prompt_template)

        expanded_queries_str = ""
        if hasattr(response, 'content'): # For AIMessage objects
            expanded_queries_str = response.content
        elif isinstance(response, str): # For direct string responses
            expanded_queries_str = response
        else:
            # Fallback if the response type is unexpected
            print(f"Warning: Unexpected LLM response type for query expansion: {type(response)}")
            expanded_queries_str = str(response)

        # print(f"DEBUG: Raw LLM output for expansion:\n'{expanded_queries_str}'") # For debugging

        # Parse the response: split by newline, filter out empty strings
        generated_queries = [q.strip() for q in expanded_queries_str.split('\n') if q.strip()]

        cleaned_queries = []
        for q in generated_queries:
            # Remove common list markers like "1. ", "- ", "* " more robustly
            q_cleaned = re.sub(r"^\s*([\d]+\.|[\-\*]+)\s*", "", q)
            if q_cleaned: # Ensure not empty after cleaning
                cleaned_queries.append(q_cleaned)

        # print(f"DEBUG: Cleaned generated queries: {cleaned_queries}") # For debugging

        # Combine with original query, ensuring original is first and no duplicates (case-insensitive)
        all_queries = [original_query]
        processed_queries_lower = {original_query.lower()} # Set for efficient duplicate checking

        for q_cleaned in cleaned_queries:
            if q_cleaned.lower() not in processed_queries_lower:
                all_queries.append(q_cleaned)
                processed_queries_lower.add(q_cleaned.lower())

        # If no new unique queries were generated beyond the original
        if len(all_queries) == 1 and cleaned_queries:
            # This can happen if LLM returns variations that are empty after cleaning or exact duplicates
            # print(f"DEBUG: No unique expansions generated. Original query: {original_query}")
            pass # Fall through to return [original_query] or list with original + unique expansions

        # print(f"INFO: Original query: '{original_query}', Expanded to: {all_queries}")
        return all_queries

    except Exception as e:
        print(f"ERROR: Error during query expansion for '{original_query}': {e}")
        return [original_query] # Fallback to original query


def rerank_documents(
    original_query: str,
    documents: list[Document],
    llm: BaseChatModel,
    top_n_to_select: int = RERANKING_TOP_N_SELECT
) -> list[Document]:
    """
    Re-ranks a list of documents based on their relevance to the original query using an LLM.
    Uses app_config default for top_n_to_select.

    Args:
        original_query: The user's original query.
        documents: A list of Langchain Document objects retrieved from vector search.
        llm: The language model instance to use for re-ranking.
        top_n_to_select: The desired number of top documents to select and return.

    Returns:
        A list of re-ranked (and potentially subsetted) Document objects.
        Returns the original list (or its subset) if re-ranking fails or no documents are provided.
    """
    if not documents:
        print("INFO: No documents provided for re-ranking.")
        return []

    formatted_docs = ""
    for i, doc in enumerate(documents):
        content_snippet = doc.page_content.replace('\n', ' ').replace('"', '\"').strip()
        content_snippet = content_snippet[:500]
        formatted_docs += f"Document {i+1}:\nContent: \"{content_snippet}...\"\n---\n"

    prompt = f"""Given the user's original query: "{original_query}"

And the following retrieved documents (with snippets):
---
{formatted_docs}---

Identify the TOP {top_n_to_select} documents from the list above that are MOST RELEVANT to the original query.
List the numbers of these top {top_n_to_select} documents in descending order of relevance (most relevant first).
Return ONLY the numbers, separated by commas. For example, if Document 3 is most relevant, then Document 1, then Document 2, return: 3,1,2
Do not include any other text, explanation, or formatting. Just the comma-separated numbers.
If you think fewer than {top_n_to_select} documents are relevant, list only those that are. If none are relevant, return an empty string or 'None'.
"""

    try:
        # print(f"DEBUG: Reranking Prompt:\n{prompt}") # For debugging
        response = llm.invoke(prompt)

        response_content = ""
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)

        # print(f"DEBUG: Raw LLM output for reranking:\n'{response_content}'") # For debugging

        if response_content.strip().lower() == "none" or not response_content.strip():
            print("INFO: LLM indicated no documents were relevant for reranking.")
            return []

        selected_indices_str = re.findall(r'\d+', response_content)

        reranked_docs_by_llm = []
        parsed_indices_from_llm = []

        for s_idx in selected_indices_str:
            try:
                doc_idx_0_based = int(s_idx) - 1
                if 0 <= doc_idx_0_based < len(documents):
                    if doc_idx_0_based not in parsed_indices_from_llm:
                        reranked_docs_by_llm.append(documents[doc_idx_0_based])
                        parsed_indices_from_llm.append(doc_idx_0_based)
                else:
                    print(f"WARNING: Reranker LLM provided out-of-bounds index: {s_idx} (parsed as {doc_idx_0_based}) for {len(documents)} docs.")
            except ValueError:
                print(f"WARNING: Reranker LLM provided non-integer value in output: {s_idx}")

        final_selected_docs = reranked_docs_by_llm

        if len(final_selected_docs) < top_n_to_select:
            for doc_idx, doc in enumerate(documents):
                if len(final_selected_docs) >= top_n_to_select:
                    break
                if doc_idx not in parsed_indices_from_llm:
                    final_selected_docs.append(doc)

        result_docs = final_selected_docs[:top_n_to_select]

        # print(f"INFO: Original docs count: {len(documents)}, Reranked and selected docs count: {len(result_docs)}")
        return result_docs

    except Exception as e:
        print(f"ERROR: Error during document re-ranking for query '{original_query}': {e}")
        return documents[:top_n_to_select]


if __name__ == '__main__':
    print("Testing retrieval_utils.py...")

    # Dummy LLM for testing the logic without making real API calls
    class DummyLLM(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompt_str = messages[0].content if messages else ""
            mock_output = ""

            if "Generate" in prompt_str and "alternative phrasings" in prompt_str: # Query Expansion
                input_query_match = re.search(r"Given the user query: '(.*?)'", prompt_str)
                num_expansions_match = re.search(r"Generate (\d+) alternative", prompt_str)
                input_query = input_query_match.group(1) if input_query_match else "test query"
                num_expansions = int(num_expansions_match.group(1)) if num_expansions_match else 2
                print(f"\n--- DummyLLM received query expansion prompt for: '{input_query}', num_expansions: {num_expansions} ---")
                if "benefits of AI in healthcare" in input_query:
                    mock_output = "AI applications in medical field\nadvantages of artificial intelligence in hospitals\nhow AI improves patient outcomes"
                else:
                    for i in range(num_expansions):
                        mock_output += f"{i+1}. Expanded query for '{input_query}' number {i+1}\n"

            elif "Identify the TOP" in prompt_str and "MOST RELEVANT" in prompt_str: # Re-ranking
                original_query_match = re.search(r"original query: \"(.*?)\"", prompt_str)
                original_query = original_query_match.group(1) if original_query_match else "test query"
                doc_count_match = len(re.findall(r"Document \d+:", prompt_str))
                top_n_match = re.search(r"TOP (\d+) documents", prompt_str)
                top_n = int(top_n_match.group(1)) if top_n_match else 2
                print(f"\n--- DummyLLM received re-ranking prompt for query: '{original_query}', docs: {doc_count_match}, top_n: {top_n} ---")
                # Simulate returning a few indices based on doc_count and top_n
                if doc_count_match >= 3 and top_n >=3: mock_output = "3,1,2"
                elif doc_count_match >= 2 and top_n >=2: mock_output = "2,1"
                elif doc_count_match >= 1 and top_n >=1: mock_output = "1"
                else: mock_output = "None" # Or empty

            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration
            return ChatGeneration(message=AIMessage(content=mock_output))

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop, run_manager, **kwargs)

        @property
        def _llm_type(self) -> str:
            return "dummy-chat-model"

    dummy_llm_instance = DummyLLM()

    print("\n--- Testing query expansion ---")
    test_queries_exp = ["benefits of AI in healthcare", "future of renewable energy"]
    for query in test_queries_exp:
        expanded = expand_query(query, dummy_llm_instance, num_expansions=2)
        print(f"Original: '{query}'\nExpanded: {expanded}\n")

    print("\n--- Testing document re-ranking ---")
    sample_docs = [
        Document(page_content="Doc 1: tentang AI dan healthcare.", metadata={"id": "d1"}),
        Document(page_content="Doc 2: AI benefits for doctors and patients.", metadata={"id": "d2"}),
        Document(page_content="Doc 3: Renewable energy is good for the planet.", metadata={"id": "d3"}),
        Document(page_content="Doc 4: AI in healthcare improves diagnostics.", metadata={"id": "d4"}),
    ]

    query_rerank = "AI in healthcare"
    reranked = rerank_documents(query_rerank, sample_docs, dummy_llm_instance, top_n_to_select=2)
    print(f"Original Query: '{query_rerank}'\nReranked Docs (expected top 2):")
    for i,doc in enumerate(reranked):
        print(f"  {i+1}. {doc.page_content[:50]}... (ID: {doc.metadata.get('id')})")
    print("")

    query_rerank_fewer = "Renewable energy"
    reranked_fewer = rerank_documents(query_rerank_fewer, [sample_docs[2]], dummy_llm_instance, top_n_to_select=2)
    print(f"Original Query: '{query_rerank_fewer}' (1 doc input, top_n=2)\nReranked Docs:")
    for i,doc in enumerate(reranked_fewer):
        print(f"  {i+1}. {doc.page_content[:50]}... (ID: {doc.metadata.get('id')})")
    print("")

    print("retrieval_utils.py test finished.")
