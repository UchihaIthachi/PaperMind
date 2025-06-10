from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage # For LLM interaction
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting llm
from langgraph.graph import StateGraph, END
import functools # For using partial to pass llm to node
import logging

from src.config.app_config import SUMMARIZER_MAX_INPUT_CHARS

logger = logging.getLogger(__name__)

# Define Summarizer State
class SummarizerState(TypedDict):
    # Input can be a list of documents or a single context string
    documents_to_summarize: Optional[List[Document]]
    context_string_to_summarize: Optional[str]
    original_query: Optional[str] # To make summary query-aware if needed

    # Output fields
    summary: str
    error: Optional[str]

# Implement summarize_node
def summarize_node(state: SummarizerState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Performs summarization based on the input documents or context string.
    """
    logger.info("--- SUMMARIZER NODE ---")
    documents = state.get("documents_to_summarize")
    context_str = state.get("context_string_to_summarize")
    query = state.get("original_query")
    text_to_summarize = ""

    if documents:
        text_to_summarize = "\n\n---\n\n".join([doc.page_content for doc in documents])
        logger.info(f"Summarizing {len(documents)} documents.")
    elif context_str:
        text_to_summarize = context_str
        logger.info("Summarizing provided context string.")
    else:
        logger.error("No content provided for summarization in summarizer_node.")
        return {"summary": "", "error": "No content provided for summarization."}

    if not text_to_summarize.strip():
        logger.error("Content provided for summarization is empty in summarizer_node.")
        return {"summary": "", "error": "Content provided for summarization is empty."}

    # Constructing messages for the chat model
    messages: List[BaseMessage] = []
    system_message_content = "You are an expert summarizer. Condense the following text into a clear, concise, and coherent summary."
    if query:
        system_message_content += f" The summary should be particularly relevant to the original query: '{query}'."
    messages.append(SystemMessage(content=system_message_content))

    # Add a limited amount of text to summarize to avoid exceeding token limits
    # This is a simple truncation; more sophisticated methods might be needed for very long texts.
    # max_summary_input_length = 10000 # Example character limit for text to summarize
    if len(text_to_summarize) > SUMMARIZER_MAX_INPUT_CHARS:
        logger.warning(f"Text to summarize exceeds {SUMMARIZER_MAX_INPUT_CHARS} chars. Truncating.")
        text_to_summarize = text_to_summarize[:SUMMARIZER_MAX_INPUT_CHARS] + "..."

    messages.append(HumanMessage(content=f"Please summarize the following text:\n\n---\n{text_to_summarize}\n---"))

    try:
        logger.info(f"Invoking LLM for summarization (query-aware: {bool(query)}).")
        response = llm.invoke(messages)
        summary = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Summarization successful. Summary length: {len(summary)}")
        return {"summary": summary.strip(), "error": None}
    except Exception as e:
        logger.error(f"Error during summarization LLM call: {e}")
        return {"summary": "", "error": str(e)}

# Create create_summarizer_graph function
def create_summarizer_graph(llm: BaseChatModel):
    """
    Creates and compiles a simple LangGraph for text summarization.

    Args:
        llm: The language model instance to use for summarization.

    Returns:
        A compiled LangGraph application.
    """
    workflow = StateGraph(SummarizerState)

    # Use functools.partial to pass the llm argument to the node function
    summarize_node_with_llm = functools.partial(summarize_node, llm=llm)

    workflow.add_node("summarize", summarize_node_with_llm)
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", END) # Simple linear graph: summarize then end

    logger.info("Summarizer graph created.")
    return workflow.compile()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Basic config for testing
    logger.info("Testing summarizer_graph.py structure...")

    # Mock LLM for structural testing
    class MockSummarizerLLM(BaseChatModel):
        def invoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
            logger.info(f"MockSummarizerLLM invoked with {len(messages)} messages.")
            # Simulate summary generation
            text_to_summarize_content = ""
            original_query_content = ""
            for msg in messages:
                if isinstance(msg, HumanMessage): # Assuming text to summarize is in HumanMessage
                    text_to_summarize_content = msg.content
                if isinstance(msg, SystemMessage) and "Original Query" in msg.content: # Basic check
                    original_query_content = "mock_query_present"

            summary_text = f"Mock summary of: '{text_to_summarize_content[:50]}...'"
            if original_query_content:
                summary_text += f" (Relevant to query: {original_query_content})"
            # Ensure AIMessage is imported or defined. For this example, assuming it's imported.
            return AIMessage(content=summary_text)

        def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Any: pass
        async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Any: pass
        @property
        def _llm_type(self) -> str: return "mock-summarizer-llm"

    mock_llm = MockSummarizerLLM()
    summarizer_app = create_summarizer_graph(mock_llm)
    logger.info("Summarizer graph compiled with Mock LLM.")

    # Test case 1: Summarize documents
    docs_to_summarize = [
        Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
        Document(page_content="It extends LangChain with the ability to coordinate multiple chains or actors over cycles of computation in a graph.")
    ]
    input_state_docs = SummarizerState(
        documents_to_summarize=docs_to_summarize,
        context_string_to_summarize=None,
        original_query="What is LangGraph?",
        summary="", error=None
    )
    result_docs = summarizer_app.invoke(input_state_docs)
    logger.info(f"\nTest 1 (Documents) Result: Summary='{result_docs.get('summary')}', Error='{result_docs.get('error')}'")

    # Test case 2: Summarize context string
    string_to_summarize = "LangGraph allows for cycles, making it suitable for agent-like behaviors where the LLM calls tools and then reasons about the tool outputs in a loop."
    input_state_string = SummarizerState(
        documents_to_summarize=None,
        context_string_to_summarize=string_to_summarize,
        original_query=None,
        summary="", error=None
    )
    result_string = summarizer_app.invoke(input_state_string)
    logger.info(f"\nTest 2 (String) Result: Summary='{result_string.get('summary')}', Error='{result_string.get('error')}'")

    # Test case 3: No input
    input_state_none = SummarizerState(
        documents_to_summarize=None,
        context_string_to_summarize=None,
        original_query=None,
        summary="", error=None
    )
    result_none = summarizer_app.invoke(input_state_none)
    logger.info(f"\nTest 3 (No Input) Result: Summary='{result_none.get('summary')}', Error='{result_none.get('error')}'")

    # Test case 4: Empty string input
    input_state_empty_str = SummarizerState(
        documents_to_summarize=None,
        context_string_to_summarize=" ",
        original_query=None,
        summary="", error=None
    )
    result_empty_str = summarizer_app.invoke(input_state_empty_str)
    logger.info(f"\nTest 4 (Empty String) Result: Summary='{result_empty_str.get('summary')}', Error='{result_empty_str.get('error')}'")

    logger.info("\nsummarizer_graph.py test finished.")
