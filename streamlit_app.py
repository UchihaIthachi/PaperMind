import streamlit as st
# from dotenv import load_dotenv # No longer needed, app_config handles it
import os # Still needed for R2 status check in sidebar (though TAVILY_API_KEY can come from config)

# Configure logging at the very start, before any other app imports if possible
import sys
import os
# Add src to path before other src imports to ensure logger can find config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Adds project root
# Assuming streamlit_app.py is in the root, and src is a subdir.
# If streamlit_app.py is in src, then '..' to get to root, then 'src' is not needed.
# Current structure is streamlit_app.py at root, src is a subdir.
# So, to import src.utils.logger, src must be in path.
# The sys.path.append in evaluation script was '..' then 'src'.
# Here, if streamlit_app.py is at root, 'src' itself is the top-level package for imports.

from src.utils.logger import initial_app_logging_config
initial_app_logging_config() # Call once to set up logging for the whole app

import logging # Now you can get loggers in this file too if needed
logger = logging.getLogger(__name__) # Get a logger for this file

# --- Refactored Imports (should come after logging config if they also use logging) ---
from src.app_logic.session_manager import initialize_session_state
from src.utils.document_parsers import extract_text_from_pdfs
from src.db_managers.vector_store_manager import process_and_store_chunks_in_chroma, add_texts_to_supabase_store
from src.db_managers.file_object_store import upload_file_to_r2
from src.tools.tool_definitions import get_all_tools
# main_react_agent_chat_prompt is for the old agent, LangGraph agent constructs its prompt internally or via create_agent_graph
# from src.prompts.agent_prompts import main_react_agent_chat_prompt
from src.config.app_config import (
    AGENT_MAX_ITERATIONS,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,
    CHROMA_SESSION_COLLECTION_NAME,
    TAVILY_API_KEY # Import for status check
)
from src.agent.graph import create_agent_graph, AgentState # Import LangGraph elements
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage # For chat history
from langchain_core.agents import AgentFinish # To check agent outcome

# from langchain.agents import AgentExecutor, create_react_agent # Old ReAct agent
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # Not used with current invoke() setup
import datetime # For metadata timestamp during PDF processing

# Define avatars (optional, or load from config)
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"

def main():
    st.set_page_config(page_title="PaperMind - LangGraph Agent", layout="wide")
    st.title("📚 PaperMind - AI Research Assistant")

    # load_dotenv() # Handled by app_config.py module import upon its first import (e.g. in session_manager)

    # Initialize session state (critical to do this early)
    initialize_session_state() # Sets up LLM, embedding models, DB clients, memory, etc.

    # Initialize LangGraph Agent (once per session)
    if "agent_graph_app" not in st.session_state:
        if st.session_state.llm: # Ensure LLM is ready
            # Tools need llm, session collections, etc. which should be in st.session_state
            # after initialize_session_state()
            tools = get_all_tools(
                llm=st.session_state.llm,
                pdf_session_collection=st.session_state.get("pdf_session_collection"), # Will be None initially
                supabase_vector_store=st.session_state.get("supabase_vector_store"),
                # tavily_api_key is now handled internally by get_all_tools using imported config
                st_embedding_model=st.session_state.get("embedding_model_st")
            )
            if tools: # Ensure tools were actually created (primarily checks if llm and st_embedding_model were available)
                st.session_state.agent_graph_app = create_agent_graph(st.session_state.llm, tools)
                st.sidebar.success("✅ LangGraph Agent initialized!")
                print("INFO: LangGraph Agent initialized successfully.")
            else:
                st.sidebar.error("⚠️ LangGraph Agent tools failed to initialize. Agent not ready.")
                print("ERROR: LangGraph Agent tools failed to initialize.")
                st.session_state.agent_graph_app = None # Explicitly set to None
        else:
            st.sidebar.error("⚠️ LLM not available. LangGraph Agent not initialized.")
            print("ERROR: LLM not available for LangGraph Agent initialization.")
            st.session_state.agent_graph_app = None


    # --- Sidebar for uploads and status ---
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF files for session and long-term memory",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if uploaded_files:
            # Accumulators for batch processing
            all_chunks_for_session_rag = []
            all_processed_chunks_for_supabase = []
            all_metadatas_for_supabase = []

            for uploaded_file_obj in uploaded_files:
                st.write(f"Processing {uploaded_file_obj.name}...")
                r2_object_key = None

                # 1. Upload to R2 (if configured)
                if st.session_state.r2_client:
                    with st.spinner(f"Uploading {uploaded_file_obj.name} to R2..."):
                        r2_object_key = upload_file_to_r2(uploaded_file_obj)
                    if r2_object_key:
                        st.success(f"☁️ Uploaded {uploaded_file_obj.name} to R2 as {r2_object_key}.")
                    else:
                        st.error(f"❌ Failed to upload {uploaded_file_obj.name} to R2.")
                else:
                    st.info("R2 storage not configured. Skipping direct R2 upload for this file.")

                # 2. Extract text
                with st.spinner(f"Extracting text from {uploaded_file_obj.name}..."):
                    # extract_text_from_pdfs expects a list of file objects
                    current_file_text = extract_text_from_pdfs([uploaded_file_obj])

                if not current_file_text:
                    st.warning(f"No text extracted from {uploaded_file_obj.name}. Skipping further processing for this file.")
                    continue

                # 3. Chunk text
                # Using RecursiveCharacterTextSplitter directly here
                from langchain.text_splitter import RecursiveCharacterTextSplitter # Ensure import
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                    separators=CHUNK_SEPARATORS,
                    length_function=len
                )
                file_specific_chunks = text_splitter.split_text(current_file_text)

                if not file_specific_chunks:
                    st.info(f"No text chunks generated from {uploaded_file_obj.name} after splitting.")
                    continue

                st.info(f"Generated {len(file_specific_chunks)} chunks from {uploaded_file_obj.name}.")
                all_chunks_for_session_rag.extend(file_specific_chunks)

                # 4. Prepare metadata for Supabase (if configured)
                if st.session_state.supabase_vector_store:
                    for i, chunk in enumerate(file_specific_chunks):
                        chunk_metadata = {
                            "source": "pdf_upload",
                            "original_file": uploaded_file_obj.name,
                            "chunk_index_in_file": i,
                            "r2_object_key": r2_object_key if r2_object_key else "N/A",
                            "upload_time": datetime.datetime.now().isoformat(),
                        }
                        all_processed_chunks_for_supabase.append(chunk)
                        all_metadatas_for_supabase.append(chunk_metadata)

            # After processing all files:
            # 5. Populate session RAG (ChromaDB)
            if all_chunks_for_session_rag and st.session_state.chromadb_client and st.session_state.embedding_model_st:
                with st.spinner("Updating session RAG store (ChromaDB)..."):
                    # Pass the initialized chromadb_client and embedding_model_st
                    _, session_collection = process_and_store_chunks_in_chroma(
                        "\n\n".join(all_chunks_for_session_rag),
                        chroma_client=st.session_state.chromadb_client,
                        embedding_model_st=st.session_state.embedding_model_st,
                        collection_name=CHROMA_SESSION_COLLECTION_NAME
                    )
                    if session_collection:
                        st.session_state.pdf_session_collection = session_collection
                        st.success(f"✅ Session RAG store updated with {len(all_chunks_for_session_rag)} total chunks.")
                    else:
                        st.warning("⚠️ Could not initialize/update session PDF store (ChromaDB).")
            elif not all_chunks_for_session_rag and uploaded_files:
                 st.info("No text chunks available from uploaded files to update session RAG store.")


            # 6. Add all collected chunks and their specific metadatas to Supabase
            if st.session_state.supabase_vector_store and all_processed_chunks_for_supabase:
                with st.spinner(f"Adding {len(all_processed_chunks_for_supabase)} total PDF chunks to long-term memory (Supabase)..."):
                    add_texts_to_supabase_store( # Renamed function in vector_store_manager
                        st.session_state.supabase_vector_store,
                        texts=all_processed_chunks_for_supabase,
                        metadatas=all_metadatas_for_supabase
                    )
            elif not all_processed_chunks_for_supabase and uploaded_files:
                 st.info("No processable text chunks from PDFs to add to long-term memory (Supabase).")
            elif uploaded_files and not st.session_state.supabase_vector_store:
                st.warning("Long-term memory (Supabase) not available. PDF content not added to it.")

        st.divider()
        st.header("Service Status")
        # Supabase Status
        if st.session_state.get("supabase_vector_store"):
            st.success("✅ Supabase Vector Store connected.", icon="🔗")
        elif st.session_state.get("supabase_client"):
             st.warning("Supabase client connected, but Vector Store failed. Check SQL setup & console.", icon="⚠️")
        else:
            st.error("Supabase not configured or connection failed. Long-term memory disabled.", icon="❌")

        # R2 Status
        if st.session_state.get("r2_client"):
            st.success("✅ R2 Original File Storage configured.", icon="☁️")
        elif not os.getenv("R2_ENDPOINT_URL") and not os.getenv("R2_ACCESS_KEY_ID") :
            st.info("R2 env vars not set. File storage disabled.")
        else:
            st.warning("R2 client init failed. File storage may be affected. Check console.", icon="⚠️")

        # Tavily Status (now using imported constant from app_config)
        if TAVILY_API_KEY: # Check the imported constant
            st.success("✅ Tavily Web Search configured.", icon="🌐")
        else:
            st.info("Tavily API key not found (via app_config). Web search tool disabled.")


    # --- Main Chat Interface ---
    # The old st.session_state.memory (ConversationBufferMemory) is replaced by LangGraph's chat_history.
    # We'll use a new session state variable, e.g., st.session_state.lg_messages, for LangGraph.
    if "lg_messages" not in st.session_state:
        st.session_state.lg_messages = [] # Stores BaseMessage objects for LangGraph

    # Display chat history
    for msg in st.session_state.lg_messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user", avatar=USER_AVATAR).write(msg.content)
        elif isinstance(msg, AIMessage):
            # AIMessage might contain tool calls which we might not want to directly render.
            # For now, just render content. If tool_calls are present, it's an intermediate step.
            if msg.content: # Only render if there's textual content
                 st.chat_message("assistant", avatar=ASSISTANT_AVATAR).write(msg.content)
            # Optionally, could render tool calls differently if desired (e.g. st.expander)
            # if msg.tool_calls:
            #    for tc in msg.tool_calls:
            #        st.chat_message("assistant", avatar=ASSISTANT_AVATAR).info(f"Tool call: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            # Optionally display tool results - can be verbose
            # with st.expander(f"Tool Result ({msg.name} - Call ID: {msg.tool_call_id})", expanded=False):
            #    st.markdown(msg.content)
            pass # Often, we don't display raw tool messages in main chat.


    user_query = st.chat_input("Ask your research question...")

    if user_query:
        st.session_state.lg_messages.append(HumanMessage(content=user_query))
        st.chat_message("user", avatar=USER_AVATAR).write(user_query)

        if st.session_state.agent_graph_app is None:
            st.error("Agent graph is not initialized. Cannot process query. Check logs and API keys.")
            st.stop()

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            message_placeholder = st.empty() # For streaming final answer or showing "Thinking..."
            message_placeholder.markdown("Thinking...")

            # Prepare input for the graph
            # LangGraph's `add_messages` in AgentState handles history accumulation.
            # We pass the current user input and the existing history.
            graph_input = {
                "input": user_query,
                # Pass the current list of BaseMessage objects
                "chat_history": st.session_state.lg_messages[:-1] # History up to the previous message
            }

            full_response_content = ""
            final_agent_finish_output = None

            try:
                # Stream events from the graph
                # The StreamlitCallbackHandler is not directly compatible with LangGraph's stream().
                # We need to manually process graph events for UI updates.
                # Consider creating a custom callback or parsing logic here.

                # For simplicity, let's use invoke() first to ensure graph runs, then adapt to stream() if time.
                # The stream() is better for showing intermediate steps/thoughts.
                # For now, with invoke(), we get the final state.

                # Using invoke to get the final state:
                final_state = st.session_state.agent_graph_app.invoke(graph_input, {"recursion_limit": AGENT_MAX_ITERATIONS})

                # Extract final response from agent_outcome or the last relevant AIMessage in chat_history
                if final_state and isinstance(final_state.get("agent_outcome"), AgentFinish):
                    final_agent_finish_output = final_state["agent_outcome"].return_values['output']
                    full_response_content = final_agent_finish_output
                elif final_state and final_state.get("chat_history"):
                    # Look for the last AIMessage that is not a tool call, as that's likely the final answer
                    for msg in reversed(final_state["chat_history"]):
                        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                            full_response_content = msg.content
                            break
                    if not full_response_content: # Fallback if no suitable AIMessage found
                        full_response_content = "Processing complete. (No explicit textual output from agent)"
                else:
                    full_response_content = "Sorry, I couldn't process your request properly."

            except Exception as e:
                full_response_content = f"An error occurred: {str(e)}"
                st.error(full_response_content)
                print(f"ERROR: LangGraph agent execution failed: {e}")

            message_placeholder.markdown(full_response_content)
            st.session_state.lg_messages.append(AIMessage(content=full_response_content))
            # No st.rerun() here as invoke() is synchronous. UI updates after this block.
            # If using stream, rerun might be needed after stream completion.

if __name__ == "__main__":
    main()
