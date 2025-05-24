import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

class LegalDraftingAssistant:
    def __init__(self, llm, embeddings, vector_store=None):
        """Initialize the legal drafting assistant with LLM, embeddings, and optional vector store."""
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.template_cache = {}  # Cache for document templates
        
    def set_vector_store(self, vector_store):
        """Set or update the vector store for RAG capabilities."""
        self.vector_store = vector_store
        
    def generate_agreement_template(self, agreement_type: str, parties: str, key_terms: str, 
                                   jurisdiction: str = "General") -> str:
        """Generate a legal agreement template based on user specifications."""
        
        # Format the input for the model
        parties_formatted = "\n".join([f"- {party.strip()}" for party in parties.split("\n") if party.strip()])
        
        # Enhance with RAG if vector store is available
        context = ""
        if self.vector_store:
            # Create a compressor for the retriever to extract most relevant information
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            # Query for relevant context from the knowledge base
            query = f"{agreement_type} {jurisdiction} {key_terms}"
            relevant_docs = compression_retriever.get_relevant_documents(query)
            
            if relevant_docs:
                context = "RELEVANT PRECEDENTS:\n" + "\n\n".join([doc.page_content for doc in relevant_docs])
        
        template = f"""
        You are an expert legal document drafter specializing in {jurisdiction} law.
        
        Generate a professional {agreement_type} with the following details:
        
        PARTIES INVOLVED:
        {parties_formatted}
        
        KEY TERMS TO INCLUDE:
        {key_terms}
        
        JURISDICTION:
        {jurisdiction}
        
        {context}
        
        Create a properly formatted legal agreement that:
        1. Includes standard sections for this type of agreement
        2. Uses clear, precise legal language
        3. Addresses all the key terms specified
        4. Is compliant with {jurisdiction} legal requirements
        5. Contains appropriate boilerplate clauses
        
        FORMAT THE DOCUMENT WITH APPROPRIATE SECTIONS, NUMBERING, AND STRUCTURE.
        """
        
        prompt = PromptTemplate(template=template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        agreement = chain.run({})
        
        return agreement
    
    def create_document_from_template(self, template_id: str, variables: Dict[str, str]) -> str:
        """Create a customized document from a template by replacing variables."""
        # Get template from cache or load it
        template = self._get_template(template_id)
        
        # Replace variables in the template
        for var_name, var_value in variables.items():
            placeholder = f"{{{{${var_name}$}}}}"
            template = template.replace(placeholder, var_value)
            
        return template
    
    def _get_template(self, template_id: str) -> str:
        """Get a template from cache or load it."""
        if template_id in self.template_cache:
            return self.template_cache[template_id]
        
        # In a real implementation, this would load from a database or file system
        # For now, we'll use some basic templates
        templates = {
            "nda": """
                NON-DISCLOSURE AGREEMENT
                
                This Non-Disclosure Agreement (the "Agreement") is entered into by and between:
                
                {{$party_a$}}, with its principal place of business at {{$address_a$}}
                
                and
                
                {{$party_b$}}, with its principal place of business at {{$address_b$}}
                
                (individually referred to as a "Party" and collectively as the "Parties")
                
                EFFECTIVE DATE: {{$effective_date$}}
                
                1. DEFINITION OF CONFIDENTIAL INFORMATION
                
                For purposes of this Agreement, "Confidential Information" means any data or information that is proprietary to either Party and not generally known to the public, whether in tangible or intangible form, including but not limited to: {{$confidential_information_definition$}}
                
                2. TERM
                
                The obligations of this Agreement shall remain in effect for a period of {{$duration$}} from the Effective Date.
                
                3. NON-DISCLOSURE
                
                The Receiving Party agrees not to use any Confidential Information for any purpose except to evaluate and engage in discussions concerning a potential business relationship between the Parties.
                
                {{$additional_clauses$}}
                
                IN WITNESS WHEREOF, the Parties hereto have executed this Agreement as of the Effective Date.
                
                {{$party_a$}}
                
                By: ________________________
                Name: {{$signatory_a$}}
                Title: {{$title_a$}}
                
                {{$party_b$}}
                
                By: ________________________
                Name: {{$signatory_b$}}
                Title: {{$title_b$}}
            """,
            # Add more templates as needed
        }
        
        if template_id in templates:
            self.template_cache[template_id] = templates[template_id]
            return templates[template_id]
        else:
            raise ValueError(f"Template '{template_id}' not found")
    
    def suggest_improvements(self, document_text: str, jurisdiction: str = "General") -> Dict[str, Any]:
        """Analyze a legal document and suggest improvements and risks."""
        
        template = f"""
        Analyze the following legal document from a {jurisdiction} legal perspective:
        
        {document_text}
        
        Provide a detailed analysis including:
        1. Potential legal risks or ambiguities
        2. Missing standard clauses or sections
        3. Language improvements for clarity and enforceability
        4. Jurisdiction-specific compliance issues
        5. Suggested additions or modifications
        
        Format your response as a structured list of recommendations.
        """
        
        prompt = PromptTemplate(template=template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        analysis = chain.run({})
        
        # Structure the output
        return {
            "analysis": analysis,
            "jurisdiction": jurisdiction,
        }
        
    def extract_legal_definitions(self, document_text: str) -> List[Dict[str, str]]:
        """Extract defined terms from a legal document."""
        
        template = f"""
        Extract all defined terms from the following legal document:
        
        {document_text}
        
        For each defined term, provide:
        1. The term itself
        2. Its definition as written in the document
        3. The section or clause number where it appears
        
        Format your response as a list, with each item containing term, definition, and location.
        """
        
        prompt = PromptTemplate(template=template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        extraction = chain.run({})
        
        # Parse the extraction into structured data
        # This is a simple implementation; in practice, you'd use more robust parsing
        definitions = []
        lines = extraction.split("\n")
        current_def = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Term:") or line.startswith("- Term:"):
                if current_def and "term" in current_def:
                    definitions.append(current_def)
                current_def = {"term": line.split(":", 1)[1].strip()}
            elif line.startswith("Definition:") or line.startswith("- Definition:"):
                if current_def:
                    current_def["definition"] = line.split(":", 1)[1].strip()
            elif line.startswith("Location:") or line.startswith("- Location:"):
                if current_def:
                    current_def["location"] = line.split(":", 1)[1].strip()
        
        # Add the last definition if it exists
        if current_def and "term" in current_def:
            definitions.append(current_def)
            
        return definitions
