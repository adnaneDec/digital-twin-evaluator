import streamlit as st
import os
import openai
import dotenv

# Only for user query caching
from functools import lru_cache

# Minimal langchain_community for doc retrieval
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from openai import OpenAIError, RateLimitError

dotenv.load_dotenv()

FAISS_PATH = ".streamlit/faiss_store"

st.title("Minimal GPT + FAISS Demo (Cached Queries)")

###############################################################################
# 1) Load the pre-built FAISS index (no repeated doc embeddings)
###############################################################################
@st.cache_resource
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(FAISS_PATH):
        st.error("No FAISS store found! Please build it first.")
        return None
    try:
        return FAISS.load_local(
            FAISS_PATH, 
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading FAISS: {e}")
        return None


vectorstore = load_vectorstore()

if not vectorstore:
    st.stop()

###############################################################################
# 2) Cache user query embeddings so repeated identical queries 
#    don’t cause more embedding calls.
###############################################################################
if "query_cache" not in st.session_state:
    st.session_state["query_cache"] = {}  # dict: {"query_string": embedding_vector}

def embed_query_with_cache(query_text: str):
    """Embed user query, caching to avoid repeated calls for same text."""
    cache = st.session_state["query_cache"]
    if query_text in cache:
        return cache[query_text]
    else:
        try:
            embedder = OpenAIEmbeddings()
            query_vec = embedder.embed_query(query_text)
            cache[query_text] = query_vec
            return query_vec
        except (RateLimitError, OpenAIError) as e:
            st.error(f"OpenAI embedding error: {e}")
            return None

###############################################################################
# 3) Minimal retrieval + Chat
###############################################################################
def answer_query(query_text: str, k=4):
    """Retrieve docs from FAISS using user query embedding, then Chat."""
    query_vec = embed_query_with_cache(query_text)
    if query_vec is None:
        return "Embedding failed. Try again later.", []

    # Use vectorstore's similarity_search_by_vector
    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(query_vec, k=k)
    except RateLimitError:
        return "Embeddings daily limit reached. Try again tomorrow.", []
    except OpenAIError as e:
        return f"OpenAI error: {e}", []

    docs = [d[0] for d in docs_and_scores]
    if not docs:
        return "No relevant documents found.", []

    # Combine doc text for context
    doc_texts = "\n\n".join([doc.page_content for doc in docs])
    system_prompt = f"""
    You have the following documents:
    {doc_texts}

    Answer the user's question, or say you don't have enough info.
    """

    # Chat
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    try:
        result = chat([SystemMessage(content=system_prompt),
                       HumanMessage(content=query_text)])
        return result.content, docs
    except RateLimitError:
        return "OpenAI Chat limit reached. Try again later.", []
    except OpenAIError as e:
        return f"OpenAI error: {e}", []

###############################################################################
# 4) Streamlit UI
###############################################################################
user_query = st.text_input("Ask about the docs:")
if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please type a query first.")
    else:
        with st.spinner("Searching..."):
            answer, retrieved_docs = answer_query(user_query)
        st.subheader("Answer:")
        st.write(answer)

        if retrieved_docs:
            with st.expander("Sources"):
                for doc in retrieved_docs:
                    st.write(doc.metadata.get("source", "Unknown"))
