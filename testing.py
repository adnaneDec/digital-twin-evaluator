import os
from glob import glob
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI  # Updated import

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

def load_pdfs_from_directory(directory_path):
    """
    Load and combine text from all PDFs in a directory.
    """
    raw_text = ""
    pdf_files = glob(os.path.join(directory_path, "*.pdf"))
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    return raw_text

def split_text(raw_text, chunk_size=1000, chunk_overlap=200):
    """
    Split raw text into smaller chunks for efficient processing.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

def create_faiss_vectorstore(text_chunks):
    """
    Create a FAISS vector store from text chunks.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(text_chunks, embeddings)

def answer_query(vectorstore, query):
    """
    Perform a similarity search on the vector store and answer the query.
    """
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = vectorstore.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

# Main workflow
if __name__ == "__main__":
    # Directory containing PDFs
    pdf_directory = "/Users/adnanedrissielbouzidi/Library/CloudStorage/GoogleDrive-adnane.drissielbouzidi@square-management.com/Mon Drive/onboarding consultant"

    # Load and process text from PDFs
    raw_text = load_pdfs_from_directory(pdf_directory)
    print("Loaded raw text from PDFs.")

    # Split text into chunks
    text_chunks = split_text(raw_text)
    print(f"Text split into {len(text_chunks)} chunks.")

    # Create FAISS vector store
    vectorstore = create_faiss_vectorstore(text_chunks)
    print("FAISS vector store created.")

    # Answer queries
    queries = [
        "Which article has these authors YASSINE QAMSANE, EFE C. BALTA, JOHN FARIS, and KIRA BARTON",
        "what are the requirements/conditions to say that a system is a digital twin",
    ]

    for query in queries:
        answer = answer_query(vectorstore, query)
        print(f"Query: {query}\nAnswer: {answer}\n")
