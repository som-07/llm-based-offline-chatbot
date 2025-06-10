
# 🧠 LocalRAG-Chatbot

A fully **local Retrieval-Augmented Generation (RAG) chatbot** that allows you to **upload documents** (PDFs, TXTs), **query** them using a **FAISS** index for retrieval, and **generate answers** with a **quantized Mistral 7B model** using `llama-cpp-python`.

⚡ Works **offline**, **private**, and is **GPU-accelerated**.

🕹 Deployed at [https://llm-chatbot-20.streamlit.app/](https://llm-chatbot-20.streamlit.app/)
---

## ✨ Features

- 🗂 Upload PDFs and text files
- 🧩 Automatic document splitting and chunking
- 🧠 Semantic search with FAISS vector database
- 🔥 Fast, local LLM generation (Mistral-7B-Instruct Q4_K_M)
- 📖 Contextual answers based strictly on uploaded documents
- ⚡ Asynchronous streaming of the generated answer
- 🔒 100% offline - your data stays on your machine

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **FAISS** (vector search)
- **Sentence Transformers** (for embeddings)
- **Llama.cpp** (for local model inference)
- **LangChain** (for document loaders & text splitting)

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Jagjeet0518/llm-chatbot.git
cd llm-chatbot
```

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Download the Model (optional)

Download a **quantized Mistral 7B model** (e.g., `mistral-7b-instruct-v0.1.Q4_K_M.gguf`) and place it in a `models/` directory.

You can get models from:
- [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

> **Note**: Quantized models are smaller and run faster on consumer GPUs.

---

## 🚀 How it Works

### Document Processing
- Upload PDF or TXT files.
- Files are loaded and **split into overlapping chunks** (chunk size: 1000 tokens, overlap: 200).
- Each chunk is **embedded** using `all-MiniLM-L6-v2` and stored into a **FAISS** index.

### Querying
- User question is embedded.
- Top-k (default: 3) relevant chunks are retrieved.
- A prompt is constructed with the chunks as context.
- The local Mistral model generates the answer **streamed back word-by-word**.

---

## 📄 Example Usage

```python
# Initialize chatbot
bot = LocalRAGChatbot()

# Load and process documents
documents = bot.load_docs(uploaded_files)
bot.process_documents(documents)

# Ask a question
async for word in bot.query_stream_async("What is the main topic of the document?"):
    print(word, end="")
```

---

## ⚡ Future Enhancements

- Add source attribution for answers
- Hybrid search (BM25 + FAISS)
- Reranking using a second LLM
- Web UI with file uploader and chat interface
- Support more file formats (.docx, .csv)
- Dynamic prompt engineering
- Conversation memory

---

## 🧩 Project Structure

```
localrag-chatbot/
│
├── models/                 # Quantized LLM files (.gguf)
├── docs/                   # Uploaded documents
├── main.py                  # Main class (LocalRAGChatbot)
├── requirements.txt        # Python dependencies
└── README.md                # This file
```

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it.

---

## 📢 Acknowledgements

- [Facebook AI](https://github.com/facebookresearch/faiss) for FAISS
- [Hugging Face](https://huggingface.co/sentence-transformers) for Sentence-Transformers
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for local LLM inference
- [LangChain](https://github.com/langchain-ai/langchain) for document handling

---
