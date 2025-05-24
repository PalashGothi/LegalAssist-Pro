
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class AgreementGenerator:
    def __init__(self, llm, retriever):
        """Initialize AgreementGenerator with LLM and retriever."""
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = PromptTemplate(
            input_variables=["agreement_type", "parties", "key_terms", "jurisdiction", "context"],
            template="""Generate a {agreement_type} for the following parties: {parties}.
            Include these key terms: {key_terms}.
            Ensure compliance with {jurisdiction} laws.
            Use the following context from similar documents: {context}
            
            Provide the agreement in a clear, professional format."""
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        self.compliance_prompt = PromptTemplate(
            input_variables=["agreement", "jurisdiction"],
            template="""Analyze the following agreement for compliance with {jurisdiction} laws:
            
            {agreement}
            
            Provide:
            1. A summary of compliance status
            2. List of potential compliance issues
            3. Suggestions for improvement"""
        )

    def generate_agreement(self, agreement_type, parties, key_terms, jurisdiction, template_mode="Standard Template", reference_text=None):
        """Generate a legal agreement based on provided parameters."""
        try:
            query = f"{agreement_type} template for {parties} with terms: {key_terms} in {jurisdiction}"
            context = ""
            
            if template_mode == "RAG-Enhanced Template":
                if reference_text:
                    context = reference_text
                else:
                    # Use retriever to get relevant context
                    result = self.qa_chain({"query": query})
                    context = result["result"] if result else ""
            else:
                # Standard template mode uses minimal context
                context = f"Standard {agreement_type} template"

            prompt = self.prompt_template.format(
                agreement_type=agreement_type,
                parties=parties,
                key_terms=key_terms,
                jurisdiction=jurisdiction,
                context=context
            )

            # Generate agreement using LLM
            agreement = self.llm(prompt)

            return agreement

        except Exception as e:
            raise Exception(f"Error generating agreement: {str(e)}")

    def check_compliance(self, agreement, jurisdiction):
        """Check the generated agreement for compliance with specified jurisdiction."""
        try:
            prompt = self.compliance_prompt.format(
                agreement=agreement,
                jurisdiction=jurisdiction
            )

            # Run compliance check using LLM
            compliance_result = self.llm(prompt)

            # Parse result into structured format (assuming LLM returns formatted text)
            # This is a simplified parsing; you may need to adjust based on LLM output
            summary = compliance_result.split("1.")[1].split("2.")[0].strip() if "1." in compliance_result else "Compliance status unclear"
            issues = compliance_result.split("2.")[1].split("3.")[0].strip().split("\n") if "2." in compliance_result else []
            suggestions = compliance_result.split("3.")[1].strip().split("\n") if "3." in compliance_result else []

            return {
                "summary": summary,
                "issues": [issue.strip() for issue in issues if issue.strip()],
                "suggestions": [suggestion.strip() for suggestion in suggestions if suggestion.strip()]
            }

        except Exception as e:
            return {
                "summary": f"Error checking compliance: {str(e)}",
                "issues": [],
                "suggestions": []
            }

