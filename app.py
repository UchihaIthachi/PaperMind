import os
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from litellm import completion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import ArxivQueryRun
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Login to HuggingFace if token is available
if huggingface_token:
    login(token=huggingface_token)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Load SentenceTransformer model
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Arxiv Tool
arxiv_tool = ArxivQueryRun()

# Utility: Extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text
    return all_text

# Utility: Process extracted text and store in ChromaDB
def process_text_and_store(all_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(all_text)

    # Recreate the knowledge base collection
    try:
        client.delete_collection(name="knowledge_base")
    except Exception:
        pass
    collection = client.create_collection(name="knowledge_base")

    for i, chunk in enumerate(chunks):
        embedding = text_embedding_model.encode(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"source": "pdf", "chunk_id": i}],
            documents=[chunk]
        )
    return collection

# Utility: Semantic search from vector database
def semantic_search(query, collection, top_k=2):
    query_embedding = text_embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

# Utility: Generate final response
def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    response = completion(
        model="gemini/gemini-1.5-flash",
        messages=[{"content": prompt, "role": "user"}],
        api_key=gemini_api_key
    )
    return response['choices'][0]['message']['content']

# Streamlit app main function
def main():
    st.set_page_config(page_title="RAG Research Assistant", layout="wide")
    st.title("📚 RAG-Powered Research Paper Assistant")

    option = st.radio("Choose an option:", ("📄 Upload PDFs", "🔍 Search arXiv"))

    if option == "📄 Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                all_text = extract_text_from_pdfs(uploaded_files)
                collection = process_text_and_store(all_text)
            st.success("✅ PDF content processed and stored successfully!")

            query = st.text_input("🔎 Enter your query about the papers:")
            if st.button("Execute Query") and query:
                with st.spinner("Fetching answer..."):
                    results = semantic_search(query, collection)
                    context = "\n".join(results['documents'][0])
                    response = generate_response(query, context)
                st.subheader("📜 Generated Answer:")
                st.write(response)

    elif option == "🔍 Search arXiv":
        search_query = st.text_input("🔎 Enter your search term for arXiv:")

        if st.button("Search ArXiv") and search_query:
            with st.spinner("Searching arXiv..."):
                arxiv_results = arxiv_tool.invoke(search_query)
                st.session_state["arxiv_text"] = arxiv_results
                collection = process_text_and_store(arxiv_results)
                st.session_state["collection"] = collection
            st.success("✅ ArXiv papers processed successfully!")
            st.subheader("🔎 Search Results")
            st.write(arxiv_results)

        if "arxiv_text" in st.session_state and "collection" in st.session_state:
            followup_query = st.text_input("📝 Ask a question based on arXiv papers:")
            if st.button("Execute Query on ArXiv") and followup_query:
                with st.spinner("Fetching answer..."):
                    results = semantic_search(followup_query, st.session_state["collection"])
                    context = "\n".join(results['documents'][0])
                    response = generate_response(followup_query, context)
                st.subheader("📜 Generated Answer:")
                st.write(response)

if __name__ == "__main__":
    main()
