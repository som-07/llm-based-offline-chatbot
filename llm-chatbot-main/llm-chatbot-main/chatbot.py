import os
import faiss
import numpy as np
from llama_cpp import Llama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class LocalRAGChatbot:
    def __init__(self):
        # Embeddings model (for RAG)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        # RAG storage
        self.texts = []
        self.text_index = None
        self.llm = self.load_llama_model()

    def load_llama_model(self):
        return Llama.from_pretrained(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        )

    def load_docs(self, uploaded_files):
        """
        Load PDF, TXT files.
        """
        documents = []
        os.makedirs("docs", exist_ok=True)

        for file in uploaded_files:
            path = os.path.join("docs", file.name)
            with open(path, "wb") as f:
                f.write(file.read())
                
            # Handle PDFs
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            # Handle plain text
            elif path.lower().endswith(".txt"):
                loader = TextLoader(path)
            else:
                # Skip unsupported formats
                continue

            documents.extend(loader.load())

        return documents

    def process_documents(self, documents):
        """
        Split loaded documents into chunks, embed, and build FAISS index.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Save text for retrieval
        self.texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.encode(self.texts, convert_to_numpy=True)

        # Build FAISS index
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings generated. Please check if the documents were loaded correctly.")

        dim = embeddings.shape[1]
        self.text_index = faiss.IndexFlatL2(dim)
        self.text_index.add(np.array(embeddings))

    def save_index(self, path: str):
        if self.text_index:
            faiss.write_index(self.text_index, path)
        else:
            raise ValueError("Index not initialized. Cannot save.")

    def load_index(self, path: str):
        if os.path.exists(path):
            self.text_index = faiss.read_index(path)
        else:
            raise FileNotFoundError(f"No FAISS index found at {path}")

    async def query_stream_async(self, question, top_k=3):
        """
        Asynchronously stream the generated answer token-by-token.
        """
        if self.text_index is None:
            yield "Please upload and process documents first."
            return
        
        try:
            # Embed question and search
            print("Question:", question)
            query_embedding = self.embedder.encode(question, convert_to_numpy=True)
            print("Query Embeddings Generated!")
            # Use distances to weigh the importance of each chunk
            distances, indices = self.text_index.search(np.array([query_embedding]), top_k)
            
            # Sort by relevance score
            chunks = [(self.texts[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
            chunks.sort(key=lambda x: x[1])  # Sort by distance (smaller is better)
            
            # Use only the most relevant chunks that fit within context window
            context = "\n---\n".join([chunk[0] for chunk in chunks[:3]])
            print("Context:\n", context)
            # Build neutral prompt
            prompt = f"""
Based on the following context, provide a concise and accurate answer to the question.
Focus only on information contained in the context. Just say "I don't know" if you don't know the answer.

Context:
{context}

Question:
{question}

Answer:"""

            # Generate from local LLM with streaming
            response = self.llm(prompt, max_tokens=300, stop=["Question:", "Context:"], stream=True, temperature=0.2)
            
            # Stream back word by word
            current_word = ""
            for chunk in response:
                token = chunk["choices"][0]["text"]
                
                # Collect tokens into words
                if token.strip() and (token.endswith(" ") or token in ".,!?;:"):
                    current_word += token
                    yield current_word
                    current_word = ""
                else:
                    current_word += token
                
                # Small delay to avoid overwhelming the event loop
                import asyncio
                await asyncio.sleep(0.01)
                
            # Yield any remaining partial word
            if current_word:
                yield current_word
                
        except Exception as e:
            yield f"Error: {str(e)}"
