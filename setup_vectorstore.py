
# setup_vectorstore.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store():
    # Ensure documents directory exists
    os.makedirs("documents", exist_ok=True)
    
    # 1. Load documents using TextLoader for .txt files
    loader = DirectoryLoader(
        "documents/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    # Check if any documents were loaded
    if not documents:
        print("Error: No .txt files found in the 'documents/' directory.")
        print("Please add at least one .txt file to the 'documents/' folder and try again.")
        return False
    
    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Check if texts were generated
    if not texts:
        print("Error: No text chunks were generated from the documents.")
        print("The documents may be empty or not readable.")
        return False
    
    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # 4. Create and save vector store
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local("vector_store")
        print("Vector store created successfully!")
        return True
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return False

if __name__ == "__main__":
    print("Ensure your legal documents (.txt files) are in the 'documents/' folder.")
    input("Press Enter when ready to create vector store...")
    success = create_vector_store()
    if success:
        print("Setup completed. You can now run the main application.")
    else:
        print("Setup failed. Please check the error messages above and try again.")
