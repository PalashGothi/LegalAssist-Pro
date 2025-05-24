
import nltk
from nltk.tokenize import sent_tokenize
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from summarizer import Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import numpy as np

# Download NLTK data
nltk.download('punkt')

class LegalSummarizer:
    """Class for summarizing legal documents with multiple techniques"""
    
    def __init__(self, llm):
        """Initialize legal summarizer with required models"""
        # Use provided LLM
        self.llm = llm
        
        # Initialize summarizers
        self.bert_model = Summarizer()
        self.text_rank_summarizer = TextRankSummarizer()
    
    def text_rank_summarize(self, text, sentences_count=5):
        """TextRank summarization implementation"""
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.text_rank_summarizer(parser.document, sentences_count)
        return " ".join(str(sentence) for sentence in summary)
    
    def ai_summarize(self, document_text, summary_type="Comprehensive"):
        """Generate AI-based summary using LLM"""
        sentences = sent_tokenize(document_text)
        text_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1000:
                current_chunk += sentence + " "
            else:
                text_chunks.append(current_chunk)
                current_chunk = sentence + " "
        
        if current_chunk:
            text_chunks.append(current_chunk)
        
        chunk_summaries = []
        for i, chunk in enumerate(text_chunks):
            template = f"""
            Analyze the following part {i+1} of a legal document:
            
            TEXT:
            {chunk}
            
            Provide a brief summary of this section, focusing on key legal points.
            """
            
            prompt = PromptTemplate(template=template, input_variables=[])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            chunk_summary = chain.run({})
            chunk_summaries.append(chunk_summary)
        
        combined_summary = "\n\n".join(chunk_summaries)
        
        if summary_type == "Comprehensive":
            template = f"""
            Based on the following summaries of different sections of a legal document:
            
            {combined_summary}
            
            Please provide:
            1. A concise but comprehensive summary of the entire document
            2. Key legal points and obligations
            3. Potential risks or considerations
            4. Important dates or deadlines mentioned
            5. Main parties involved and their roles
            """
        elif summary_type == "Executive":
            template = f"""
            Based on the following summaries of different sections of a legal document:
            
            {combined_summary}
            
            Please provide an executive summary of this document for a busy professional, highlighting:
            1. The overall purpose in 1-2 sentences
            2. The 3-4 most important provisions
            3. Any critical deadlines or dates
            4. Key financial obligations, if any
            5. Recommended actions
            
            Keep the summary concise and business-focused.
            """
        else:  # Key Legal Points Only
            template = f"""
            Based on the following summaries of different sections of a legal document:
            
            {combined_summary}
            
            Extract and list only the key legal points from this document:
            1. Core legal obligations
            2. Rights granted or reserved
            3. Liabilities assumed or disclaimed
            4. Conditions precedent or subsequent
            5. Termination provisions
            
            Format as a bulleted list of legal points only.
            """
        
        prompt = PromptTemplate(template=template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run({})
    
    def traditional_summarize(self, document_text, summary_type="Comprehensive"):
        """Generate summary using traditional NLP techniques"""
        try:
            summary = ""
            
            if summary_type == "Comprehensive":
                if self.bert_model:
                    try:
                        input_text = document_text[:10000] if len(document_text) > 10000 else document_text
                        summary = self.bert_model(input_text, ratio=0.3)
                    except Exception as e:
                        print(f"BERT summarization failed: {str(e)}")
                
                if not summary:
                    try:
                        if len(document_text) > 10000:
                            chunks = [document_text[i:i+10000] for i in range(0, len(document_text), 10000)]
                            summary_parts = []
                            for chunk in chunks:
                                summary_part = self.text_rank_summarize(chunk)
                                if summary_part:
                                    summary_parts.append(summary_part)
                            summary = " ".join(summary_parts)
                        else:
                            summary = self.text_rank_summarize(document_text)
                    except Exception as e:
                        print(f"TextRank summarization failed: {str(e)}")
                
                if not summary:
                    sentences = sent_tokenize(document_text)
                    legal_keywords = ["shall", "agree", "obligation", "right", "liability", 
                                     "term", "condition", "represent", "warrant", "terminate"]
                    
                    sentence_scores = []
                    for sentence in sentences:
                        score = sum(1 for keyword in legal_keywords if keyword.lower() in sentence.lower())
                        sentence_scores.append((sentence, score))
                    
                    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:10]
                    sentence_position = {sent: i for i, sent in enumerate(sentences)}
                    ordered_top_sentences = sorted(
                        [s[0] for s in top_sentences], 
                        key=lambda s: sentence_position.get(s, 0)
                    )
                    summary = " ".join(ordered_top_sentences)
            
            elif summary_type == "Executive":
                paragraphs = document_text.split("\n\n")
                extracted_paragraphs = [paragraphs[0]]
                
                key_terms = ["purpose", "objective", "summary", "conclusion", 
                            "payment", "term", "deadline", "obligation"]
                
                for para in paragraphs[1:]:
                    if any(term in para.lower() for term in key_terms):
                        extracted_paragraphs.append(para)
                
                if len(extracted_paragraphs) > 4:
                    extracted_paragraphs = [extracted_paragraphs[0]] + extracted_paragraphs[1:4]
                
                summary = "\n\n".join(extracted_paragraphs)
            
            else:  # Key Legal Points Only
                sentences = sent_tokenize(document_text)
                legal_terms = ["shall", "must", "required", "obligation", "agreement", 
                              "consent", "right", "liability", "warranty", "indemnify", 
                              "terminate", "breach", "enforce", "void", "null"]
                
                legal_points = []
                for sentence in sentences:
                    if any(term in sentence.lower() for term in legal_terms):
                        legal_points.append(f"• {sentence}")
                
                if len(legal_points) > 15:
                    legal_points = legal_points[:15]
                
                summary = "\n".join(legal_points)
            
            return summary if summary else "Traditional summarization methods failed to extract a meaningful summary."
        
        except Exception as e:
            return f"Error generating traditional summary: {str(e)}"
    
    def hybrid_summarize(self, document_text, summary_type="Comprehensive"):
        """Generate a hybrid summary combining AI and traditional approaches"""
        try:
            ai_summary = self.ai_summarize(document_text, summary_type)
            traditional_summary = self.traditional_summarize(document_text, summary_type)
            
            template = f"""
            I have two different summaries of the same legal document.
            
            AI-GENERATED SUMMARY:
            {ai_summary}
            
            TRADITIONAL NLP SUMMARY:
            {traditional_summary}
            
            Please create a hybrid summary that combines the strengths of both approaches.
            Focus on:
            1. Keeping the structured organization from the AI summary
            2. Including any additional factual details from the traditional summary
            3. Resolving any contradictions between the two summaries
            4. Ensuring all key legal points are included
            
            Provide this as a {summary_type.lower()} summary format.
            """
            
            prompt = PromptTemplate(template=template, input_variables=[])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run({})
        
        except Exception as e:
            return f"Error generating hybrid summary: {str(e)}"
