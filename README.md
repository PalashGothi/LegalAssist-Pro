LegalAssist-Pro
LegalAssist-Pro is an AI-powered web application designed to streamline legal document processing. It leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to provide advanced features for document summarization, legal drafting, multilingual processing, document Q&A, benchmarking, and system performance monitoring. Built with Streamlit, LangChain, and Hugging Face Transformers, it supports legal professionals by offering efficient, accurate, and multilingual document handling.
Table of Contents

Features
Prerequisites
Installation
Usage
Project Structure
Contributing
License
Contact

Features

Document Analysis & Summarization: Generate concise summaries of legal documents using AI-based, traditional NLP, or hybrid methods.
Legal Drafting: Create context-aware legal documents (e.g., NDAs, contracts) with customizable templates and jurisdiction support.
Multilingual Processing: Translate and process legal texts in languages like English, Hindi, and more, with legal term preservation.
Document Q&A: Ask questions about uploaded documents, powered by a conversational retrieval chain.
Benchmarking: Evaluate system performance with metrics like summarization accuracy and translation quality.
Dashboard: Visualize usage analytics, system performance, and recent activity logs.
Ethical Compliance: Ensures data privacy and adherence to legal standards during processing.

Prerequisites

Python: Version 3.10 or higher.
Git: For cloning the repository.
Virtual Environment: Recommended to isolate dependencies.
Vector Store: A pre-built FAISS vector store (vector_store/index.faiss) is required for document retrieval.
Dependencies: Listed in requirements.txt.

Installation

Clone the Repository:
git clone https://github.com/PalashGothi/LegalAssist-Pro.git
cd LegalAssist-Pro


Set Up a Virtual Environment:
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux


Install Dependencies:
pip install -r requirements.txt

Note: The requirements.txt includes dependencies like streamlit, langchain, transformers, faiss-cpu, and others. Ensure you have sufficient disk space and memory, as some packages (e.g., torch) are large.

Set Up the Vector Store:The application requires a FAISS vector store for document retrieval. If you don’t have a vector_store/index.faiss file:

Run the setup_vectorstore.py script (if provided) to generate it, or
Contact the repository owner for the vector store files.Place the vector_store directory in the project root.


Configure Environment Variables:Create a .env file in the project root with necessary configurations (e.g., API keys for Hugging Face, if used):
HUGGINGFACEHUB_API_TOKEN=your_token_here

Use python-dotenv to load these variables (already included in requirements.txt).


Usage

Run the Application:
streamlit run app.py

This launches the Streamlit app, accessible at http://localhost:8501 in your browser.

Log In:

Use the default admin credentials (if unchanged):
Username: admin
Password: admin123


Register a new user via the "Register" tab if needed.


Navigate Features:

Document Analysis: Upload a PDF, TXT, or DOCX file to summarize or compare summarization methods.
Legal Drafting: Select an agreement type, input parties and terms, and generate a draft.
Multilingual Processing: Translate or process texts in supported languages.
Document Q&A: Ask questions about uploaded documents.
Benchmarking: Run performance tests (admin-only).
Dashboard: View usage analytics and system metrics.


Troubleshooting:

If you encounter a "Vector store not found" error, ensure vector_store/index.faiss exists.
If the Q&A feature fails, verify that setup_vectorstore.py has been run to initialize the conversational chain.
Check the console for dependency-related errors and ensure all packages are installed correctly.



Project Structure
LegalAssist-Pro/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
├── .gitignore               # Excludes venv, cache, and other unnecessary files
├── .env                     # Environment variables (not tracked)
├── modules/                 # Custom modules for processing
│   ├── document_processor.py
│   ├── legal_summarizer.py
│   ├── multilingual_processor.py
│   ├── agreement_generator.py
│   ├── auth.py
│   ├── performance_metrics.py
│   ├── comparative_analysis.py
├── data/                    # Stores feedback and benchmark results
├── vector_store/            # FAISS vector store (not tracked)
└── README.md                # This file

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, contact:

Palash Gothi: palash.gothi@example.com
GitHub Issues: https://github.com/PalashGothi/LegalAssist-Pro/issues


Note: The virtual environment (venv) is excluded from version control to avoid large file issues. Always use requirements.txt to recreate the environment. If you need the vector store or have issues setting up, open an issue on the repository.
