# System Architecture

This document outlines the architecture for the RAG-Powered Research Paper Assistant, which has been refactored into a modular structure and utilizes LangGraph for its core agent logic.

## 1. Core Architecture (Post-Refactor with LangGraph & Summarizer)

The system is built using Python, with core logic organized within a `src/` directory and the main application entry point being `streamlit_app.py`.

*   **User Interface (`streamlit_app.py`):**
    *   **Streamlit:** Used for building the interactive web application.
    *   Manages session state initialization via `src.app_logic.session_manager`.

*   **Codebase Structure (`src/` directory):**
    *   **`agent/`**: Contains LangGraph definitions:
        *   `graph.py`: The main agent graph.
        *   `summarizer_graph.py`: The summarization subgraph.
    *   **`app_logic/`**: Session state management (`session_manager.py`).
    *   **`config/`**: Centralized configurations (`app_config.py`).
    *   **`db_managers/`**: Modules for R2, Supabase, and ChromaDB interactions.
    *   **`prompts/`**: Agent and LLM prompt templates.
    *   **`tools/`**: Definitions of tools available to the main agent.
    *   **`utils/`**: Utilities for document parsing, LLM/embedding models, and retrieval enhancements.

*   **Core Agent Logic (`src/agent/graph.py`):**
    *   **Main LangGraph Agent:**
        *   **State (`AgentState`):** Manages `input`, `chat_history`, `agent_outcome`, and new fields for summarization flow: `retrieved_tool_context`, `summarized_tool_context`, `summarization_error`.
        *   **Nodes:**
            *   **Agent Node:** Invokes the LLM (Gemini) with tools bound. Based on the current state (including potentially summarized context from previous tool calls), it decides to call a tool (`AgentAction`) or formulate a final response (`AgentFinish`). It adds the LLM's decision (AIMessage with tool calls or content) to `chat_history`. It also clears context fields (`retrieved_tool_context`, `summarized_tool_context`) for the next turn.
            *   **Tool Node:** Executes actions from the agent node using `ToolExecutor`. Results (as `ToolMessage` objects) are added to `chat_history`. The combined string content of tool outputs is stored in `retrieved_tool_context`.
            *   **Summarizer Node (`call_summarizer_node_logic`):** Invokes the summarization subgraph if called.
        *   **Summarization Subgraph (`src/agent/summarizer_graph.py`):**
            *   A separate, simple LangGraph workflow designed for text condensation.
            *   **State (`SummarizerState`):** Takes `documents_to_summarize` or `context_string_to_summarize` and an optional `original_query`. Outputs `summary` or `error`.
            *   **Node (`summarize_node`):** Uses an LLM to perform the summarization, potentially making it query-aware.
            *   This subgraph is called by the main agent graph's "summarizer" node.
        *   **Conditional Edges:**
            *   After agent decision: Route to "tools" or `END`.
            *   After tool execution: Route to "summarizer" if `len(retrieved_tool_context)` exceeds `TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION` (from `app_config`); otherwise, route directly back to "agent".
            *   After summarization: Route back to "agent".
        *   Includes max iteration limits.
    *   **LLM (`src/utils/llm_utils.py`):** Google's `gemini-1.5-flash` model is used for all LLM tasks: agent reasoning, tool usage, query expansion, re-ranking, and summarization.
    *   **Conversational Memory:** Managed by LangGraph via `chat_history` in `AgentState`.
    *   **Prompting (`src/prompts/agent_prompts.py`):** The main agent's system prompt (`SYSTEM_PROMPT_LANGGRAPH`) has been updated to guide the LLM on how to interpret and use context that might be provided by the system (i.e., the `AIMessage` containing tool outputs or summaries) before making its next decision.

*   **Data Storage, Processing, Retrieval Pipeline, Tools, Environment:** (Remain as previously described, but now tools are orchestrated by the LangGraph agent, and the retrieval pipeline's output can be summarized).

## 2. Envisioned Future Enhancements
*   (As previously listed)

## 3. Diagram (Conceptual - Refactored Architecture with LangGraph and Summarizer)

```mermaid
graph TD
    User[User] <--> UI[streamlit_app.py UI]

    subgraph ApplicationCore [src/*]
        direction TB
        SessionMgr[Session Management (src/app_logic/session_manager.py)]

        subgraph AgentExecutionGraph [Main Agent Graph (src/agent/graph.py)]
            direction LR
            AgentDecisionNode[Agent Node (LLM + Prompts)]
            ToolExecutionNode[Tool Node (Executes Tools)]
            ConditionalSummarizeEdge{Should Summarize Context?}
            SummarizerCallNode[Call Summarizer Subgraph]
        end

        SummarizerSubgraph[Summarization Subgraph (src/agent/summarizer_graph.py)]

        subgraph AgentSupportModules
            direction LR
            Prompts[Prompts (src/prompts)]
            Tools[Tool Definitions (src/tools)]
            LLM_Utils[LLM/Embedding Utilities (src/utils/llm_utils.py)]
            RetrievalUtils[Retrieval Utilities (src/utils/retrieval_utils.py)]
        end

        subgraph DataManagement [src/db_managers/*]
            direction LR
            R2StoreManager[File Object Store (R2)]
            VectorStoreManagers[Vector Store Managers (Supabase, Chroma)]
            DocParsers[Document Parsers (src/utils/document_parsers.py)]
        end
    end

    UI -- User Input / Uploads --> SessionMgr
    SessionMgr -- Initializes/Provides State & Config --> AgentDecisionNode
    SessionMgr -- Manages PDF Ingestion --> IngestionPipelineApp[PDF Ingestion (in streamlit_app.py)]

    AgentDecisionNode -- Requests Tool(s) --> ToolExecutionNode
    ToolExecutionNode -- Tool Output --> ConditionalSummarizeEdge
    ConditionalSummarizeEdge -- Yes (Context too long) --> SummarizerCallNode
    ConditionalSummarizeEdge -- No (Context manageable) --> AgentDecisionNode
    SummarizerCallNode -- Invokes --> SummarizerSubgraph
    SummarizerSubgraph -- Summarized Context --> AgentDecisionNode

    AgentDecisionNode -- Uses --> LLM_Utils
    AgentDecisionNode -- Uses --> Prompts
    Tools -- Called by --> ToolExecutionNode
    Tools -- Uses --> LLM_Utils
    Tools -- Uses --> RetrievalUtils
    Tools -- Uses --> VectorStoreManagers
    SummarizerSubgraph -- Uses --> LLM_Utils # Summarizer uses an LLM

    IngestionPipelineApp -- Uses --> DocParsers
    IngestionPipelineApp -- Uses --> R2StoreManager
    IngestionPipelineApp -- Uses --> VectorStoreManagers
    IngestionPipelineApp -- Uses --> LLM_Utils

    AgentDecisionNode -- Final Agent Response --> UI

    subgraph ExternalServices
        SupabaseDB[(Supabase/pgvector)]
        R2ObjectStore[(Cloudflare R2)]
        SessionChromaDB[(ChromaDB)]
        ArXivAPI[ArXiv API]
        TavilyAPI[Tavily API]
        GoogleGeminiAPI[Google Gemini API]
    end

    VectorStoreManagers -- CRUD --> SupabaseDB
    VectorStoreManagers -- CRUD --> SessionChromaDB
    R2StoreManager -- CRUD --> R2ObjectStore
    Tools -- API Call --> ArXivAPI
    Tools -- API Call --> TavilyAPI
    LLM_Utils -- API Call --> GoogleGeminiAPI
```

This `architecture.md` will be updated as the project evolves.
