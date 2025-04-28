
# 📚 PaperMind - A RAG-Powered Research Paper Assistant

**PaperMind** is a lightweight research assistant that lets you:
- Upload your own PDF papers and query them
- Search academic papers from arXiv
- Ask natural language questions and get AI-powered answers

---

## 🛠️ Built With:
- **Streamlit** (UI framework)
- **LangChain** (RAG pipeline)
- **ChromaDB** (Vector database)
- **Sentence Transformers** (Embeddings)
- **Gemini API** (via LiteLLM)
- **HuggingFace models** (alternative LLM backend)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/UchihaIthachi/PaperMind.git
cd PaperMind
```

---

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Paste the following inside `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

🔗 **Where to get API Keys**:
- [Gemini API Key](https://aistudio.google.com/app/apikey)
- [HuggingFace Token](https://huggingface.co/settings/tokens)

---

### 3. Set Up Python Virtual Environment

If you don't have `venv`, install it first:

```bash
sudo apt install python3-venv
```

Then create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📚 Key Features

- 📄 Upload PDFs and extract information
- 🔍 Search research papers via ArXiv
- 💬 Ask natural language questions with RAG-based retrieval
- ⚡ Fast embedding & querying via ChromaDB + Sentence Transformers
- 🔒 Local and API-based LLM support (Gemini / HuggingFace)

---