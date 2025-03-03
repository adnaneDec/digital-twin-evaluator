from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key="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")

# Test FAISS vector store
try:
    vectorstore = FAISS.from_texts(["test document"], embeddings)
    print("FAISS module works correctly!")
except Exception as e:
    print(f"Error: {e}")
