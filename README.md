# 📚 PaperMind - Your Conversational AI Research Assistant

**PaperMind** is an intelligent research assistant featuring a sophisticated agent built with **LangGraph**. Its modular codebase, organized within a `src/` directory, allows for robust and maintainable conversational flows. Key capabilities include:
- Uploading PDF papers, which are stored persistently in Cloudflare R2. Their content is then processed for advanced RAG, available for both the current session and long-term recall.
- Searching academic papers from arXiv.
- Performing web searches for up-to-date information (via Tavily API).
- Maintaining conversational context using LangGraph's state management.
- Leveraging a persistent long-term memory (Supabase/pgvector) to recall information from previously processed documents across sessions, with metadata linking to original PDFs in R2.

The agent intelligently routes queries and employs advanced retrieval strategies, including LLM-based **query expansion**, **LLM-based re-ranking** of search results, and **automatic summarization** of lengthy tool outputs to provide focused context for relevant and accurate answers.

---

## 🛠️ Technical Stack & Architecture
- **UI Framework**: Streamlit (`streamlit_app.py` as the entry point).
- **Core Agent & Orchestration**: LangGraph, LangChain (for agent framework, RAG, tool integration, memory, summarization).
- **Codebase Structure**: Modular design with core logic in `src/` (modules for agent, prompts, tools, DB managers, utilities, app logic).
- **LLM**: Gemini API (via `langchain-google-genai`) for agent reasoning, query expansion, re-ranking, and summarization.
- **Session RAG**: ChromaDB (vector database).
- **Long-Term Memory**: Supabase (PostgreSQL with pgvector for text chunks & metadata).
- **Original Document Storage**: Cloudflare R2 (via `boto3`).
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`).
- **External Information**: ArXiv API, Tavily Search API.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/UchihaIthachi/PaperMind.git
cd PaperMind
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root by copying `src/.env.example` (if it moves there) or `.env.example`.
For now, assuming `.env.example` is at the root:
```bash
cp .env.example .env
```
Then, open `.env` and add your API keys/credentials:
```env
# Core Functionality
GEMINI_API_KEY=your_gemini_api_key

# Optional Features
HUGGINGFACE_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key

# Long-Term Memory (Supabase - for text chunks and metadata)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Original File Storage (Cloudflare R2 - for uploaded PDFs)
R2_ENDPOINT_URL=your_r2_endpoint_url
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_r2_bucket_name
```
🔗 **API Keys & Credentials**:
- Gemini: [Google AI Studio](https://aistudio.google.com/app/apikey) (Used for the agent and all LLM-driven enhancements)
- HuggingFace: [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Tavily: [Tavily AI](https://tavily.com/)
- Supabase: [Supabase Dashboard](https://supabase.com/)
- Cloudflare R2: [Cloudflare Dashboard](https://dash.cloudflare.com/) & [R2 Docs](https://developers.cloudflare.com/r2/)

### 3. Set Up Python Virtual Environment & Dependencies
(Standard venv and `pip install -r requirements.txt` instructions remain the same)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the App
The main application is now `streamlit_app.py`:
```bash
streamlit run streamlit_app.py
```
Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

---

## 🌟 Key Features

- **🤖 Advanced Conversational Agent (LangGraph)**: Utilizes LangGraph for sophisticated, stateful agent execution, managing complex conversational flows and tool orchestration.
- **🧩 Modular Codebase**: Core logic organized within a `src/` directory for better maintainability and scalability.
- **🔍 Advanced Retrieval Strategies**:
    - Employs LLM-based **query expansion** to broaden search coverage.
    - Uses LLM-based **re-ranking** to refine and prioritize retrieved documents.
    - **Automatic Context Summarization**: Condenses lengthy tool outputs using an LLM to provide focused context to the agent, improving efficiency.
- **📄 Session PDF RAG**: Upload research papers for immediate querying within the current session using ChromaDB.
- **🧠 Persistent Long-Term Memory**: Remembers information from ingested documents across sessions (optional, via Supabase/pgvector).
- **🗄️ Original Document Storage**: Stores uploaded PDF files in Cloudflare R2 for persistence, linked from long-term memory metadata (optional).
- **学术 ArXiv Integration**: Search academic papers on ArXiv.
- **🌐 Web Search (Optional)**: General web search via Tavily API if configured.

---
## 📖 Project Structure & Usage

The core application logic is now organized within the `src/` directory, containing modules for:
- `agent`: LangGraph agent definitions, including the main agent and summarization subgraph.
- `app_logic`: High-level application flow and session management.
- `db_managers`: Interactions with R2, Supabase, and ChromaDB.
- `prompts`: Agent and LLM prompt templates.
- `tools`: Definitions of tools available to the agent.
- `utils`: Helper utilities for document parsing, LLM interactions, and retrieval enhancements.

The main Streamlit UI is managed by `streamlit_app.py`.

**Usage:**
1.  **Configure Services (Optional but Recommended)**: Set up API keys in `.env`.
2.  **Upload PDFs**: Use the sidebar. PDFs are stored in R2 (if configured), and their content processed for session RAG (ChromaDB) and long-term memory (Supabase, if configured).
3.  **Ask Questions**: The LangGraph agent processes your query. If tools retrieve extensive information, it may be automatically summarized before the agent formulates a final answer.
4.  **Conversational Follow-up**: The agent maintains context throughout your session.

---

## 🏛️ System Architecture

For a detailed explanation of the system's components, data flow (now including the LangGraph agent structure with its summarization subgraph, query expansion, and re-ranking steps), please see the [System Architecture Document](./architecture.md).

---
