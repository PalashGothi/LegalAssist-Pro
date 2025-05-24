
import os
import docx2txt
from PyPDF2 import PdfReader
from typing import Tuple, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline

class DocumentProcessor:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

    def process_document(self, file_path: str) -> Tuple[Optional[any], str, Optional[any]]:
        """Returns (vectordb, text, conversation) tuple"""
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            text = ""
            
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif ext == ".pdf":
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif ext == ".docx":
                text = docx2txt.process(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Placeholder for conversation chain
            # Note: app.py expects a conversation object, so we return a dummy chain
            conversation = None  # Replace with actual chain if needed
            
            return None, text, conversation
        
        except Exception as e:
            raise ValueError(f"Document processing failed: {str(e)}")
