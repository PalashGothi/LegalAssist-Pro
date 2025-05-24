
import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import json
import hashlib
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="LegalAssist Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "LegalAssist Pro - Advanced Legal Document Assistant"
    }
)

# Import custom modules
from modules.document_processor import DocumentProcessor
from modules.legal_summarizer import LegalSummarizer
from modules.multilingual_processor import MultilingualProcessor
from modules.agreement_generator import AgreementGenerator
from modules.auth import authenticate_user, create_user, hash_password, load_users
from modules.performance_metrics import track_interaction, load_metrics, calculate_performance
from modules.comparative_analysis import compare_summarization_techniques, compare_human_ai_documents

# Load environment variables
load_dotenv()

# Initialize LLM
@st.cache_resource
def load_llm():
    try:
        # Load model and tokenizer for google/flan-t5-base
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create a pipeline for text2text-generation
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=768,
            temperature=0.1,
            device=-1  # Use CPU; set to 0 for GPU if available
        )
        
        # Wrap in HuggingFacePipeline for LangChain
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}. Please ensure 'transformers' and 'torch' are installed and you have sufficient memory.")
        st.stop()

# Initialize Vector Store
@st.cache_resource
def load_vector_store():
    if not os.path.exists("vector_store/index.faiss"):
        st.error("Vector store not found. Please run setup_vectorstore.py first")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "role" not in st.session_state:
        st.session_state.role = ""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = ""
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = ""
    if "traditional_summary" not in st.session_state:
        st.session_state.traditional_summary = ""
    if "ai_summary" not in st.session_state:
        st.session_state.ai_summary = ""
    if "feedback_collected" not in st.session_state:
        st.session_state.feedback_collected = {}
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = []

# Initialize components
init_session_state()
llm = load_llm()
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever()

# Authentication function
def login_page():
    st.title("LegalAssist Pro - Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            success, token = authenticate_user(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                # Load user role
                users = load_users()
                st.session_state.role = users.get(username, {}).get("role", "user")
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        if st.session_state.role == "admin" or not st.session_state.authenticated:
            new_username = st.text_input("New Username", key="reg_username")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            role = st.selectbox("Role", ["user", "admin"], index=0)
            
            if st.button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if create_user(new_username, new_password, role):
                        st.success("User registered successfully! You can now login.")
                    else:
                        st.error("Username already exists")
        else:
            st.info("Only administrators can create new accounts")

def document_analysis_page(doc_processor, legal_summarizer):
    st.header("Document Analysis & Summarization")
    
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "txt", "docx"])
    
    if uploaded_file:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.document_text = ""
            st.session_state.document_summary = ""
            st.session_state.traditional_summary = ""
            st.session_state.ai_summary = ""
            
            with st.spinner("Processing document..."):
                try:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the document
                    st.session_state.vectordb, st.session_state.document_text, st.session_state.conversation = doc_processor.process_document(tmp_path)
                    
                    # Remove temporary file
                    os.unlink(tmp_path)
                    
                    st.success("Document processed successfully!")
                    
                    # Track interaction
                    track_interaction(
                        action="document_upload",
                        document_name=uploaded_file.name,
                        timestamp=datetime.now().isoformat()
                    )
                
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    return
    
    if st.session_state.document_text:
        st.subheader("Document Content")
        with st.expander("View Document Text"):
            st.text_area("Document Content", st.session_state.document_text, height=200)
        
        col1, col2 = st.columns(2)
        
        with col1:
            summarization_options = st.multiselect(
                "Select summarization methods to compare:",
                ["AI-based Summarization", "Traditional NLP Summarization", "Hybrid Approach"],
                ["AI-based Summarization", "Traditional NLP Summarization"]
            )
            
        with col2:
            summary_type = st.selectbox(
                "Summary Type",
                ["Comprehensive", "Executive", "Key Legal Points Only"]
            )
        
        if st.button("Generate Summaries"):
            with st.spinner("Analyzing document..."):
                try:
                    if "AI-based Summarization" in summarization_options:
                        try:
                            st.session_state.ai_summary = legal_summarizer.ai_summarize(st.session_state.document_text, summary_type)
                        except Exception as e:
                            st.error(f"AI Summarization failed: {str(e)}. Please ensure the LLM is properly configured.")
                    
                    if "Traditional NLP Summarization" in summarization_options:
                        st.session_state.traditional_summary = legal_summarizer.traditional_summarize(st.session_state.document_text, summary_type)
                    
                    if "Hybrid Approach" in summarization_options:
                        hybrid_summary = legal_summarizer.hybrid_summarize(st.session_state.document_text, summary_type)
                        st.session_state.hybrid_summary = hybrid_summary
                    
                    # Track interaction
                    track_interaction(
                        action="generate_summary",
                        document_name=st.session_state.uploaded_file_name,
                        summary_type=summary_type,
                        methods=summarization_options,
                        timestamp=datetime.now().isoformat()
                    )
                
                except Exception as e:
                    st.error(f"Error analyzing document: {str(e)}")
        
        # Display summaries if available
        if "AI-based Summarization" in summarization_options and st.session_state.ai_summary:
            with st.expander("AI-based Summary", expanded=True):
                st.write(st.session_state.ai_summary)
                st.download_button(
                    label="Download AI Summary",
                    data=st.session_state.ai_summary,
                    file_name="ai_summary.txt",
                    mime="text/plain"
                )
        
        if "Traditional NLP Summarization" in summarization_options and st.session_state.traditional_summary:
            with st.expander("Traditional NLP Summary", expanded=True):
                st.write(st.session_state.traditional_summary)
                st.download_button(
                    label="Download Traditional Summary",
                    data=st.session_state.traditional_summary,
                    file_name="traditional_summary.txt",
                    mime="text/plain"
                )
        
        if "Hybrid Approach" in summarization_options and hasattr(st.session_state, 'hybrid_summary'):
            with st.expander("Hybrid Summary", expanded=True):
                st.write(st.session_state.hybrid_summary)
                st.download_button(
                    label="Download Hybrid Summary",
                    data=st.session_state.hybrid_summary,
                    file_name="hybrid_summary.txt",
                    mime="text/plain"
                )
        
        # Compare summaries if multiple are generated
        if (st.session_state.ai_summary and st.session_state.traditional_summary) or \
           (st.session_state.ai_summary and hasattr(st.session_state, 'hybrid_summary')) or \
           (st.session_state.traditional_summary and hasattr(st.session_state, 'hybrid_summary')):
            st.subheader("Comparative Analysis")
            
            if st.button("Compare Summarization Techniques"):
                with st.spinner("Comparing summaries..."):
                    try:
                        comparison_results = compare_summarization_techniques(
                            document=st.session_state.document_text,
                            ai_summary=st.session_state.ai_summary if "AI-based Summarization" in summarization_options else None,
                            traditional_summary=st.session_state.traditional_summary if "Traditional NLP Summarization" in summarization_options else None,
                            hybrid_summary=st.session_state.hybrid_summary if "Hybrid Approach" in summarization_options and hasattr(st.session_state, 'hybrid_summary') else None
                        )
                        
                        st.subheader("Comparison Results")
                        
                        # Display metrics
                        metrics_cols = st.columns(len(comparison_results["metrics"]))
                        
                        for i, (metric_name, value) in enumerate(comparison_results["metrics"].items()):
                            metrics_cols[i].metric(metric_name, f"{value:.2f}")
                        
                        # Display qualitative analysis
                        st.write("### Qualitative Analysis")
                        st.write(comparison_results["qualitative_analysis"])
                        
                        # Display visualization
                        st.write("### Visualization")
                        st.pyplot(comparison_results["visualization"])
                    
                    except Exception as e:
                        st.error(f"Error comparing summaries: {str(e)}")
        
        # Feedback collection
        st.subheader("Provide Feedback")
        summary_quality = st.slider("Summary Quality (1-5)", 1, 5, 3)
        accuracy_rating = st.slider("Legal Accuracy (1-5)", 1, 5, 3)
        feedback_text = st.text_area("Additional Feedback", "")
        
        if st.button("Submit Feedback"):
            feedback_id = f"{st.session_state.username}_{datetime.now().isoformat()}"
            st.session_state.feedback_collected[feedback_id] = {
                "username": st.session_state.username,
                "document": st.session_state.uploaded_file_name,
                "summary_quality": summary_quality,
                "accuracy_rating": accuracy_rating,
                "feedback_text": feedback_text,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save feedback to file
            os.makedirs("data", exist_ok=True)
            with open("data/feedback.json", "w") as f:
                json.dump(list(st.session_state.feedback_collected.values()), f)
            
            st.success("Thank you for your feedback!")

def legal_drafting_page(agreement_generator):
    st.header("Legal Document Drafting")
    
    agreement_type = st.selectbox(
        "Select agreement type:",
        ["Non-Disclosure Agreement (NDA)", "Service Agreement", "Employment Contract", 
         "Sales Contract", "Lease Agreement", "Partnership Agreement", "Customized Agreement"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        parties = st.text_area("Enter parties involved (one per line):", height=100)
        key_terms = st.text_area("Enter key terms and conditions to include:", height=150)
    
    with col2:
        jurisdiction = st.selectbox(
            "Select Jurisdiction",
            ["India", "United States", "United Kingdom", "European Union", "International"]
        )
        
        template_mode = st.radio(
            "Template Mode",
            ["Standard Template", "RAG-Enhanced Template"]
        )
        
        # Reference document upload (optional, for RAG enhancement)
        reference_doc = None
        if template_mode == "RAG-Enhanced Template":
            reference_doc = st.file_uploader("Upload reference document (optional)", type=["pdf", "txt", "docx"])
            st.info("RAG-Enhanced templates use similar documents as reference to improve accuracy and relevance")
    
    if st.button("Generate Agreement"):
        with st.spinner("Generating agreement..."):
            try:
                # Process reference document if provided
                reference_text = None
                if reference_doc is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{reference_doc.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(reference_doc.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract text from reference document
                    doc_processor = DocumentProcessor()
                    _, reference_text, _ = doc_processor.process_document(tmp_path)
                    os.unlink(tmp_path)
                
                # Generate agreement
                agreement = agreement_generator.generate_agreement(
                    agreement_type=agreement_type,
                    parties=parties,
                    key_terms=key_terms,
                    jurisdiction=jurisdiction,
                    template_mode=template_mode,
                    reference_text=reference_text
                )
                
                st.subheader(f"Generated {agreement_type}:")
                st.text_area("Agreement Text:", value=agreement, height=400)
                
                # Display compliance check if available
                compliance_report = agreement_generator.check_compliance(agreement, jurisdiction)
                
                with st.expander("Compliance Report"):
                    st.write(compliance_report["summary"])
                    
                    # Display compliance issues if any
                    if compliance_report["issues"]:
                        st.warning("Potential Compliance Issues:")
                        for issue in compliance_report["issues"]:
                            st.write(f"- {issue}")
                    else:
                        st.success("No compliance issues detected")
                    
                    # Display suggestions if any
                    if compliance_report["suggestions"]:
                        st.info("Suggestions for Improvement:")
                        for suggestion in compliance_report["suggestions"]:
                            st.write(f"- {suggestion}")
                
                # Option to download the agreement
                st.download_button(
                    label="Download Agreement",
                    data=agreement,
                    file_name=f"{agreement_type.lower().replace(' ', '_')}.txt",
                    mime="text/plain"
                )
                
                # Track interaction
                track_interaction(
                    action="generate_agreement",
                    agreement_type=agreement_type,
                    template_mode=template_mode,
                    jurisdiction=jurisdiction,
                    timestamp=datetime.now().isoformat()
                )
            
            except Exception as e:
                st.error(f"Error generating agreement: {str(e)}")

def multilingual_page(multilingual_processor):
    st.header("Multilingual Legal Processing")
    
    # Language selection
    source_language = st.selectbox(
        "Source Language",
        ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"]
    )
    
    target_language = st.selectbox(
        "Target Language",
        ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"]
    )
    
    # Input method
    input_method = st.radio(
        "Input Method",
        ["Text Input", "Document Upload"]
    )
    
    input_text = ""
    
    if input_method == "Text Input":
        input_text = st.text_area("Enter text to translate:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload document to translate", type=["pdf", "txt", "docx"])
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract text from document
                    doc_processor = DocumentProcessor()
                    _, input_text, _ = doc_processor.process_document(tmp_path)
                    
                    # Remove temporary file
                    os.unlink(tmp_path)
                    
                    st.success("Document processed for translation!")
                
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    processing_options = st.multiselect(
        "Processing Options",
        ["Translation", "Legal Terms Preservation", "Format Preservation", "Cultural Adaptation"],
        ["Translation", "Legal Terms Preservation"]
    )
    
    if st.button("Process") and input_text:
        with st.spinner(f"Processing from {source_language} to {target_language}..."):
            try:
                result = multilingual_processor.process_text(
                    text=input_text,
                    source_language=source_language,
                    target_language=target_language,
                    options=processing_options
                )
                
                st.subheader("Processed Result:")
                st.text_area("Result:", value=result["translated_text"], height=300)
                
                # Display preserved legal terms
                if "Legal Terms Preservation" in processing_options and result["preserved_terms"]:
                    with st.expander("Preserved Legal Terms"):
                        for term in result["preserved_terms"]:
                            st.write(f"- **{term['original']}**: {term['translated']}")
                
                # Display legal accuracy report
                with st.expander("Legal Accuracy Report"):
                    st.write(result["accuracy_report"])
                
                # Option to download the translated document
                st.download_button(
                    label="Download Translation",
                    data=result["translated_text"],
                    file_name=f"translation_{target_language.lower()}.txt",
                    mime="text/plain"
                )
                
                # Track interaction
                track_interaction(
                    action="multilingual_processing",
                    source_language=source_language,
                    target_language=target_language,
                    options=processing_options,
                    timestamp=datetime.now().isoformat()
                )
            
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

def document_qa_page(doc_processor):
    st.header("Document Q&A")
    
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "txt", "docx"])
    
    if uploaded_file:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            
            with st.spinner("Processing document..."):
                try:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the document
                    st.session_state.vectordb, st.session_state.document_text, st.session_state.conversation = doc_processor.process_document(tmp_path)
                    
                    # Remove temporary file
                    os.unlink(tmp_path)
                    
                    st.success("Document processed! You can now ask questions.")
                
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    if st.session_state.conversation is not None:
        # Show document context
        with st.expander("Document Context"):
            st.text_area("Document Content", st.session_state.document_text[:1000] + "..." if len(st.session_state.document_text) > 1000 else st.session_state.document_text, height=150)
        
        user_question = st.text_input("Ask a question about the document:")
        
        if user_question:
            with st.spinner("Searching for answer..."):
                try:
                    # Use the ConversationalRetrievalChain to get answer
                    response = st.session_state.conversation({"question": user_question})
                    answer = response["answer"]
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    # Show sources if available
                    if "source_documents" in response:
                        with st.expander("Sources"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content)
                                st.write("---")
                    
                    # Track interaction
                    track_interaction(
                        action="document_qa",
                        document_name=st.session_state.uploaded_file_name,
                        question=user_question,
                        timestamp=datetime.now().isoformat()
                    )
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

def benchmarking_page(legal_summarizer):
    st.header("Benchmarking & Evaluation")
    
    if st.session_state.role != "admin":
        st.warning("Only administrators can access the full benchmarking features")
    
    tab1, tab2 = st.tabs(["Performance Evaluation", "Human vs AI Comparison"])
    
    with tab1:
        st.subheader("AI System Performance Evaluation")
        
        benchmark_options = st.multiselect(
            "Select benchmark tests to run:",
            ["Summarization Accuracy", "Legal Term Extraction", "Translation Quality", "Agreement Compliance"],
            ["Summarization Accuracy", "Legal Term Extraction"]
        )
        
        uploaded_file = st.file_uploader("Upload benchmark document", type=["pdf", "txt", "docx"])
        
        if uploaded_file and st.button("Run Benchmark Tests"):
            with st.spinner("Running benchmarks..."):
                try:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract text from document
                    doc_processor = DocumentProcessor()
                    _, document_text, _ = doc_processor.process_document(tmp_path)
                    
                    # Remove temporary file
                    os.unlink(tmp_path)
                    
                    # Run benchmarks
                    benchmark_results = {}
                    
                    if "Summarization Accuracy" in benchmark_options:
                        ai_summary = legal_summarizer.ai_summarize(document_text, "Comprehensive")
                        traditional_summary = legal_summarizer.traditional_summarize(document_text, "Comprehensive")
                        
                        # Calculate metrics
                        from modules.performance_metrics import evaluate_summary
                        summary_metrics = evaluate_summary(document_text, ai_summary)
                        
                        benchmark_results["Summarization"] = {
                            "AI Summary": ai_summary,
                            "Traditional Summary": traditional_summary,
                            "Metrics": summary_metrics
                        }
                    
                    if "Legal Term Extraction" in benchmark_options:
                        from modules.performance_metrics import evaluate_term_extraction
                        term_metrics = evaluate_term_extraction(document_text)
                        
                        benchmark_results["Term Extraction"] = term_metrics
                    
                    if "Translation Quality" in benchmark_options and st.session_state.role == "admin":
                        multilingual_processor = MultilingualProcessor(llm=llm)
                        translation_result = multilingual_processor.process_text(
                            text=document_text[:1000],  # Use first 1000 chars for translation benchmark
                            source_language="English",
                            target_language="Hindi",
                            options=["Translation", "Legal Terms Preservation"]
                        )
                        
                        # Re-translate back to English to evaluate quality
                        back_translation = multilingual_processor.process_text(
                            text=translation_result["translated_text"],
                            source_language="Hindi",
                            target_language="English",
                            options=["Translation"]
                        )
                        
                        from modules.performance_metrics import evaluate_translation
                        translation_metrics = evaluate_translation(
                            original_text=document_text[:1000],
                            back_translated_text=back_translation["translated_text"]
                        )
                        
                        benchmark_results["Translation"] = {
                            "Original": document_text[:1000],
                            "Translated": translation_result["translated_text"],
                            "Back-Translated": back_translation["translated_text"],
                            "Metrics": translation_metrics
                        }
                    
                    if "Agreement Compliance" in benchmark_options and st.session_state.role == "admin":
                        agreement_generator = AgreementGenerator(llm=llm, retriever=retriever)
                        agreement = agreement_generator.generate_agreement(
                            agreement_type="Non-Disclosure Agreement (NDA)",
                            parties="Company A\nCompany B",
                            key_terms="Confidential information protection\nTerm: 2 years\nGoverning law: Delhi, India",
                            jurisdiction="India",
                            template_mode="Standard Template"
                        )
                        
                        compliance_report = agreement_generator.check_compliance(agreement, "India")
                        
                        benchmark_results["Compliance"] = {
                            "Agreement": agreement,
                            "Compliance Report": compliance_report
                        }
                    
                    # Display benchmark results
                    st.subheader("Benchmark Results")
                    
                    if "Summarization" in benchmark_results:
                        with st.expander("Summarization Benchmarks", expanded=True):
                            st.write("### Metrics")
                            metrics = benchmark_results["Summarization"]["Metrics"]
                            
                            cols = st.columns(len(metrics))
                            for i, (metric, value) in enumerate(metrics.items()):
                                cols[i].metric(metric, f"{value:.2f}")
                            
                            st.write("### AI Summary")
                            st.write(benchmark_results["Summarization"]["AI Summary"])
                            
                            st.write("### Traditional Summary")
                            st.write(benchmark_results["Summarization"]["Traditional Summary"])
                    
                    if "Term Extraction" in benchmark_results:
                        with st.expander("Legal Term Extraction Benchmarks"):
                            st.write("### Metrics")
                            metrics = benchmark_results["Term Extraction"]["metrics"]
                            
                            cols = st.columns(len(metrics))
                            for i, (metric, value) in enumerate(metrics.items()):
                                cols[i].metric(metric, f"{value:.2f}")
                            
                            st.write("### Extracted Terms")
                            st.write(", ".join(benchmark_results["Term Extraction"]["terms"]))
                    
                    if "Translation" in benchmark_results:
                        with st.expander("Translation Benchmarks"):
                            st.write("### Metrics")
                            metrics = benchmark_results["Translation"]["Metrics"]
                            
                            cols = st.columns(len(metrics))
                            for i, (metric, value) in enumerate(metrics.items()):
                                cols[i].metric(metric, f"{value:.2f}")
                            
                            st.write("### Original Text")
                            st.write(benchmark_results["Translation"]["Original"])
                            
                            st.write("### Translated Text (Hindi)")
                            st.write(benchmark_results["Translation"]["Translated"])
                            
                            st.write("### Back-Translated Text")
                            st.write(benchmark_results["Translation"]["Back-Translated"])
                    
                    if "Compliance" in benchmark_results:
                        with st.expander("Agreement Compliance Benchmarks"):
                            st.write("### Compliance Report")
                            st.write(benchmark_results["Compliance"]["Compliance Report"]["summary"])
                            
                            if benchmark_results["Compliance"]["Compliance Report"]["issues"]:
                                st.warning("Issues:")
                                for issue in benchmark_results["Compliance"]["Compliance Report"]["issues"]:
                                    st.write(f"- {issue}")
                            
                            if benchmark_results["Compliance"]["Compliance Report"]["suggestions"]:
                                st.info("Suggestions:")
                                for suggestion in benchmark_results["Compliance"]["Compliance Report"]["suggestions"]:
                                    st.write(f"- {suggestion}")
                    
                    # Save benchmark results
                    timestamp = datetime.now().isoformat()
                    st.session_state.benchmark_results.append({
                        "timestamp": timestamp,
                        "username": st.session_state.username,
                        "tests": benchmark_options,
                        "document": uploaded_file.name,
                        "results": {k: {key: value for key, value in v.items() if key != "AI Summary" and key != "Agreement" and key != "Translated"} 
                                  for k, v in benchmark_results.items()}
                    })
                    
                    # Save to file
                    os.makedirs("data", exist_ok=True)
                    with open("data/benchmark_results.json", "w") as f:
                        json.dump(st.session_state.benchmark_results, f)
                
                except Exception as e:
                    st.error(f"Error during benchmarking: {str(e)}")
    
    with tab2:
        st.subheader("Human vs AI Document Comparison")
        
        # Upload human-generated documents for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            human_doc = st.file_uploader("Upload human-generated document", type=["pdf", "txt", "docx"])
        
        with col2:
            ai_doc = st.file_uploader("Upload AI-generated document", type=["pdf", "txt", "docx"])
        
        if human_doc and ai_doc and st.button("Compare Documents"):
            with st.spinner("Analyzing documents..."):
                try:
                    # Process documents
                    doc_processor = DocumentProcessor()
                    
                    # Human document
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{human_doc.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(human_doc.getvalue())
                        tmp_path = tmp_file.name
                    _, human_text, _ = doc_processor.process_document(tmp_path)
                    os.unlink(tmp_path)
                    
                    # AI document
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ai_doc.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(ai_doc.getvalue())
                        tmp_path = tmp_file.name
                    _, ai_text, _ = doc_processor.process_document(tmp_path)
                    os.unlink(tmp_path)
                    
                    # Compare documents
                    comparison_results = compare_human_ai_documents(human_text, ai_text)
                    
                    # Display results
                    st.subheader("Comparison Results")
                    
                    # Metrics
                    st.write("### Similarity Metrics")
                    metrics_cols = st.columns(len(comparison_results["metrics"]))
                    for i, (metric, value) in enumerate(comparison_results["metrics"].items()):
                        metrics_cols[i].metric(metric, f"{value:.2f}")
                    
                    # Analysis summary
                    st.write("### Analysis")
                    st.write(comparison_results["analysis"])
                    
                    # Key differences visualization
                    st.write("### Key Differences")
                    st.dataframe(comparison_results["differences"])
                    
                    # Visualization
                    st.write("### Visualization")
                    st.pyplot(comparison_results["visualization"])
                
                except Exception as e:
                    st.error(f"Error comparing documents: {str(e)}")

def dashboard_page():
    st.header("System Performance Dashboard")
    
    # Load metrics data
    metrics_data = load_metrics()
    
    if not metrics_data or len(metrics_data) == 0:
        st.info("No performance data available yet.")
        return
    
    # Prepare dataframes
    interaction_df = pd.DataFrame([
        {
            "timestamp": datetime.fromisoformat(entry["timestamp"]),
            "action": entry["action"],
            **{k: v for k, v in entry.items() if k not in ["timestamp", "action"]}
        }
        for entry in metrics_data if "action" in entry
    ])
    
    # Only show if we have data
    if not interaction_df.empty:
        # Time filter
        col1, col2 = st.columns(2)
        with col1:
            time_filter = st.selectbox(
                "Time Period",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
                index=3
            )
        
        with col2:
            if st.session_state.role == "admin":
                # Check if 'username' column exists and has valid data
                username_options = ["All Users"]
                if "username" in interaction_df.columns and not interaction_df["username"].isna().all():
                    username_options += list(interaction_df["username"].unique())
                user_filter = st.multiselect(
                    "Filter by Users",
                    username_options,
                    ["All Users"]
                )
            else:
                user_filter = [st.session_state.username]
                st.info(f"Showing data for {st.session_state.username}")
        
        # Apply time filter
        now = datetime.now()
        if time_filter == "Last 24 Hours":
            filtered_df = interaction_df[interaction_df["timestamp"] > (now - pd.Timedelta(days=1))]
        elif time_filter == "Last 7 Days":
            filtered_df = interaction_df[interaction_df["timestamp"] > (now - pd.Timedelta(days=7))]
        elif time_filter == "Last 30 Days":
            filtered_df = interaction_df[interaction_df["timestamp"] > (now - pd.Timedelta(days=30))]
        else:
            filtered_df = interaction_df
        
        # Apply user filter
        if "All Users" not in user_filter and st.session_state.role == "admin" and "username" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["username"].isin(user_filter)]
        elif st.session_state.role != "admin" and "username" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["username"] == st.session_state.username]
        
        # Dashboard metrics
        st.subheader("Usage Overview")
        
        metric_cols = st.columns(4)
        
        # Calculate statistics
        total_interactions = len(filtered_df)
        unique_users = filtered_df["username"].nunique() if "username" in filtered_df.columns else 1
        doc_uploads = len(filtered_df[filtered_df["action"] == "document_upload"])
        
        if "accuracy_rating" in filtered_df.columns:
            avg_accuracy = filtered_df["accuracy_rating"].mean() if not filtered_df["accuracy_rating"].isna().all() else 0
        else:
            avg_accuracy = 0
        
        # Display metrics
        metric_cols[0].metric("Total Interactions", total_interactions)
        metric_cols[1].metric("Unique Users", unique_users)
        metric_cols[2].metric("Document Uploads", doc_uploads)
        metric_cols[3].metric("Avg. Accuracy Rating", f"{avg_accuracy:.1f}/5" if avg_accuracy > 0 else "N/A")
        
        # Charts & Visualizations
        st.subheader("Usage Analytics")
        
        # Get top actions
        action_counts = filtered_df["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Actions Distribution")
            fig = px.pie(action_counts, values="Count", names="Action", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Usage Over Time")
            # Resample by day
            filtered_df["date"] = filtered_df["timestamp"].dt.date
            usage_over_time = filtered_df.groupby("date").size().reset_index()
            usage_over_time.columns = ["Date", "Count"]
            
            fig = px.line(usage_over_time, x="Date", y="Count", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed user activity
        if st.session_state.role == "admin" and "username" in filtered_df.columns:
            st.subheader("User Activity")
            
            user_activity = filtered_df.groupby("username").size().reset_index()
            user_activity.columns = ["Username", "Actions"]
            user_activity = user_activity.sort_values("Actions", ascending=False)
            
            st.dataframe(user_activity)
        
        # System performance metrics
        st.subheader("System Performance")
        
        # Calculate performance metrics
        performance_metrics = calculate_performance(filtered_df)
        
        perf_cols = st.columns(len(performance_metrics))
        
        for i, (metric, value) in enumerate(performance_metrics.items()):
            perf_cols[i].metric(
                metric, 
                f"{value:.2f}" if isinstance(value, float) else value
            )
        
        # Recent activity log
        st.subheader("Recent Activity Log")
        
        recent_activity = filtered_df.sort_values("timestamp", ascending=False).head(10)
        
        for _, activity in recent_activity.iterrows():
            timestamp = activity["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            action = activity["action"]
            
            # Format additional details, handling array-like values
            details = ", ".join([f"{k}: {v if not isinstance(v, (list, tuple, set)) else ', '.join(map(str, v))}" 
                                 for k, v in activity.items() 
                                 if k not in ["timestamp", "action", "date"] and v is not None])
            
            st.write(f"**{timestamp}** - {action}" + (f" ({details})" if details else ""))

def main():
    # Check if user is authenticated
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    legal_summarizer = LegalSummarizer(llm=llm)
    multilingual_processor = MultilingualProcessor(llm=llm)
    agreement_generator = AgreementGenerator(llm=llm, retriever=retriever)
    
    # Sidebar
    with st.sidebar:
        st.title("LegalAssist Pro")
        st.write(f"Logged in as: **{st.session_state.username}** ({st.session_state.role})")
        
        st.subheader("Navigation")
        app_mode = st.radio(
            "Select Function",
            ["Document Analysis", "Legal Drafting", "Multilingual Processing", 
             "Document Q&A", "Benchmarking", "Dashboard"]
        )
        
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.authenticated = False
            st.rerun()
    
    # Main content based on selected mode
    if app_mode == "Document Analysis":
        document_analysis_page(doc_processor, legal_summarizer)
    elif app_mode == "Legal Drafting":
        legal_drafting_page(agreement_generator)
    elif app_mode == "Multilingual Processing":
        multilingual_page(multilingual_processor)
    elif app_mode == "Document Q&A":
        document_qa_page(doc_processor)
    elif app_mode == "Benchmarking":
        benchmarking_page(legal_summarizer)
    elif app_mode == "Dashboard":
        dashboard_page()

# Run the app
if __name__ == "__main__":
    # Create necessary folders
    os.makedirs("data", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)
    
    # Run main app
    main()
