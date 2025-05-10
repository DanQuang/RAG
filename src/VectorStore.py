import os
import tempfile
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS


def prepareVectorStore(pdf_path, embedding):
    '''
    Loads a PDF file, splits its content into manageable text chunks, and creates a FAISS vector store for efficient similarity search.

    Parameters:
    - pdf_path (str): Temporary file path of the uploaded PDF.
    - embeddings (object): Pre-trained embedding model used to convert text into numerical vectors.

    Returns:
    - vector_store (FAISS): A FAISS vector store containing the embedded text chunks for similarity search.
    '''
    
    # Load the PDF and split it into text chunks for better processing
    loader = PyPDFLoader(file_path=pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum number of characters per chunk
        chunk_overlap=200  # Overlap between chunks to preserve context
    )
    documents = loader.load_and_split(text_splitter=text_splitter)
    
    # Convert the text chunks into embeddings and store them in a FAISS vector store
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding
    )
    
    return vector_store

def getVectorStore(uploaded_file, embedding):
    if uploaded_file:
            # Save file to tmp path
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                    
                    vector_store = prepareVectorStore(file_path, embedding)
                    
                return vector_store