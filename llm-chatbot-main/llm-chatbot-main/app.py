import streamlit as st
from chatbot import LocalRAGChatbot
import asyncio

# Page config
st.set_page_config(page_title="📄 Custom RAG Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Local RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
.chat-bubble {
    max-width: 80%;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 16px;
    font-size: 16px;
    line-height: 1.5;
    display: inline-block;
}

.chat-bubble.user {
    background-color: #dcf8c6;
    color: #000;
    align-self: flex-end;
    margin-left: auto;
}

.chat-bubble.bot {
    background-color: #f1f0f0;
    color: #000;
    align-self: flex-start;
    margin-right: auto;
}

.typing {
    display: flex;
    align-items: center;
}

.typing span {
    width: 8px;
    height: 8px;
    margin: 0 2px;
    background-color: #999;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
}

.typing span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes blink {
    0%, 80%, 100% {
        opacity: 0;
    }
    40% {
        opacity: 1;
    }
}
</style>
""", unsafe_allow_html=True)

# Init chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = LocalRAGChatbot()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Upload documents
with st.sidebar:
    st.header("📎 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Reading and indexing documents..."):
            docs = st.session_state.chatbot.load_docs(uploaded_files)
            st.session_state.chatbot.process_documents(docs)
            st.success("✅ Documents processed!")


# Chat handler
async def get_chatbot_response(user_query, top_k=3):
    partial_response = ""
    bot_container = st.container()

    # Loader
    with bot_container:
        placeholder = st.markdown("""
            <div class='chat-bubble bot'>
                <div class='typing'>
                    <span></span><span></span><span></span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Stream response
    async for chunk in st.session_state.chatbot.query_stream_async(user_query, top_k):
        partial_response += chunk
        placeholder.markdown(f"<div class='chat-bubble bot'>{partial_response}▌</div>", unsafe_allow_html=True)

    placeholder.markdown(f"<div class='chat-bubble bot'>{partial_response}</div>", unsafe_allow_html=True)
    return partial_response


# Render chat history
for role, msg in st.session_state.chat_history:
    bubble_class = "user" if role == "user" else "bot"
    st.markdown(f"<div class='chat-bubble {bubble_class}'>{msg}</div>", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask something about your documents...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.markdown(f"<div class='chat-bubble user'>{user_input}</div>", unsafe_allow_html=True)
    response = asyncio.run(get_chatbot_response(user_input))
    st.session_state.chat_history.append(("bot", response))
