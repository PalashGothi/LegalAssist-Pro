import os
import json
import datetime
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import nltk
from nltk.tokenize import sent_tokenize
from rouge import Rouge

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BenchmarkEvaluator:
    """Evaluate AI-generated legal content against human-generated ground truth."""
    
    def __init__(self, llm, embeddings):
        """Initialize the benchmark evaluator."""
        self.llm = llm
        self.embeddings = embeddings
        self.rouge = Rouge()
        
        # Create directories for storing benchmarks and results
        self.benchmark_dir = Path("./benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("./evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self._init_metrics_tracking()
    
    def _init_metrics_tracking(self):
        """Initialize metrics tracking."""
        metrics_path = self.results_dir / "metrics_history.json"
        if not metrics_path.exists():
            # Create initial metrics structure
            metrics = {
                "summarization": {
                    "rouge_1": [],
                    "rouge_2": [],
                    "rouge_l": [],
                    "semantic_similarity": [],
                    "legal_accuracy": []
                },
                "drafting": {
                    "completeness": [],
                    "legal_compliance": [],
                    "semantic_similarity": []
                },
                "translation": {
                    "accuracy": [],
                    "terminology_correctness": [],
                    "cultural_appropriateness": []
                }
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def add_benchmark_document(self, document_text: str, summary: str = None, 
                              document_type: str = "agreement", 
                              metadata: Dict = None) -> str:
        """Add a benchmark document with human-generated summary."""
        # Generate a unique ID for the benchmark
        benchmark_id = str(uuid.uuid4())
        
        benchmark = {
            "id": benchmark_id,
            "document_text": document_text,
            "summary": summary,
            "document_type": document_type,
            "added_at": datetime.datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save the benchmark
        benchmark_path = self.benchmark_dir / f"{benchmark_id}.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
            
        return benchmark_id
    
    def evaluate_summary(self, document_id: str, generated_summary: str) -> Dict:
        """Evaluate a generated summary against the human benchmark."""
        # Load the benchmark document
        benchmark_path = self.benchmark_dir / f"{document_id}.json"
        if not benchmark_path.exists():
            raise ValueError(f"Benchmark with ID {document_id} not found")
            
        with open(benchmark_path, 'r') as f:
            benchmark = json.load(f)
            
        if not benchmark.get("summary"):
            raise ValueError(f"Benchmark with ID {document_id} does not have a reference summary")
        
        reference_summary = benchmark["summary"]
        document_text = benchmark["document_text"]
        
        # Calculate ROUGE scores
        try:
            rouge_scores = self.rouge.get_scores(generated_summary, reference_summary)[0]
        except Exception as e:
            rouge_scores = {
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            }
        
        # Calculate semantic similarity using embeddings
        try:
            ref_embedding = self.embeddings.embed_query(reference_summary)
            gen_embedding = self.embeddings.embed_query(generated_summary)
            
            # Convert to numpy arrays and reshape for cosine_similarity
            ref_embedding_np = np.array(ref_embedding).reshape(1, -1)
            gen_embedding_np = np.array(gen_embedding).reshape(1, -1)
            
            semantic_similarity = float(cosine_similarity(ref_embedding_np, gen_embedding_np)[0][0])
        except Exception as e:
            semantic_similarity = 0
        
        # Use LLM to evaluate legal accuracy
        template = f"""
        You are a legal expert tasked with evaluating a summary of a legal document.
        
        Original document excerpt:
        {document_text[:2000]}... [document continues]
        
        Reference (human-generated) summary:
        {reference_summary}
        
        AI-generated summary to evaluate:
        {generated_summary}
        
        Please evaluate the AI-generated summary on its legal accuracy compared to the reference summary.
        Focus on:
        1. Correctness of legal terminology
        2. Inclusion of key legal provisions
        3. Accuracy of party obligations and rights
        4. Legal nuance preservation
        
        Rate the legal accuracy on a scale of 0 to 10, where 10 is perfect accuracy.
        Provide a brief explanation for your rating.
        
        Output the rating as "Rating: X/10" followed by your explanation.
        """
        
        prompt = PromptTemplate(template=template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        legal_evaluation = chain.run({})
        
        # Extract rating from LLM response
        try:
            rating_match = re.search(r"Rating:\s*(\d+(?:\.\d+)?)/10", legal_evaluation)
            legal_accuracy = float(rating_match.group(1)) / 10 if rating_match else 0.5
        except:
            legal_accuracy = 0.5
        
        # Prepare results
        results = {
            "benchmark_id": document_id,
            "evaluation_time": datetime.datetime.now().isoformat(),
            "rouge_1_f": rouge_scores["rouge-1"]["f"],
            "rouge_2_f": rouge_scores["rouge-2"]["f"],
            "rouge_l_f": rouge_scores["rouge-l"]["f"],
            "semantic_similarity": semantic_similarity,
            "legal_accuracy": legal_accuracy,
            "legal_evaluation": legal_evaluation,
            "generated_summary": generated_summary,
            "reference_summary": reference_summary
        }
        
        # Save the evaluation results
        results_path = self.results_dir / f"summary_eval_{document_id}_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update metrics history
        self._update_metrics("summarization", {
            "rouge_1": rouge_scores["rouge-1"]["f"],
            "rouge_2": rouge_scores["rouge-2"]["f"],
            "rouge_l": rouge_scores["rouge-l"]["f"],
            "semantic_similarity": semantic_similarity,
            "legal_accuracy": legal_accuracy
        })
        
        return results
    
    def evaluate_document_draft(self, template: str, generated_draft: str, 
                               ground_truth: str) -> Dict:
        """Evaluate a generated legal document draft against a ground truth document."""
        # Use LLM to evaluate legal completeness
        completeness_template = f"""
        You are a legal expert tasked with evaluating the completeness of a generated legal document.
        
        Template/Requirements:
        {template}
        
        Ground truth (human-drafted) document:
        {ground_truth}
        
        AI-generated document to evaluate:
        {generated_draft}
        
        Please evaluate the AI-generated document on its completeness compared to the ground truth.
        Focus on:
        1. Inclusion of all required sections and clauses
        2. Coverage of all stipulated requirements
        3. Structural integrity and organization
        
        Rate the completeness on a scale of 0 to 10, where 10 is perfectly complete.
        Provide a brief explanation for your rating.
        
        Output the rating as "Rating: X/10" followed by your explanation.
        """
        
        prompt = PromptTemplate(template=completeness_template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        completeness_evaluation = chain.run({})
        
        # Use LLM to evaluate legal compliance
        compliance_template = f"""
        You are a legal expert tasked with evaluating the legal compliance of a generated legal document.
        
        Template/Requirements:
        {template}
        
        Ground truth (human-drafted) document:
        {ground_truth}
        
        AI-generated document to evaluate:
        {generated_draft}
        
        Please evaluate the AI-generated document on its legal compliance and enforceability.
        Focus on:
        1. Correct use of legal terminology
        2. Legal validity of clauses
        3. Absence of contradictory terms
        4. Proper legal structure and formatting
        
        Rate the legal compliance on a scale of 0 to 10, where 10 is perfect compliance.
        Provide a brief explanation for your rating.
        
        Output the rating as "Rating: X/10" followed by your explanation.
        """
        
        prompt = PromptTemplate(template=compliance_template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        compliance_evaluation = chain.run({})
        
        # Calculate semantic similarity
        try:
            truth_embedding = self.embeddings.embed_query(ground_truth)
            draft_embedding = self.embeddings.embed_query(generated_draft)
            
            # Convert to numpy arrays and reshape for cosine_similarity
            truth_embedding_np = np.array(truth_embedding).reshape(1, -1)
            draft_embedding_np = np.array(draft_embedding).reshape(1, -1)
            
            semantic_similarity = float(cosine_similarity(truth_embedding_np, draft_embedding_np)[0][0])
        except Exception as e:
            semantic_similarity = 0
        
        # Extract ratings from LLM responses
        try:
            completeness_match = re.search(r"Rating:\s*(\d+(?:\.\d+)?)/10", completeness_evaluation)
            completeness = float(completeness_match.group(1)) / 10 if completeness_match else 0.5
        except:
            completeness = 0.5
            
        try:
            compliance_match = re.search(r"Rating:\s*(\d+(?:\.\d+)?)/10", compliance_evaluation)
            compliance = float(compliance_match.group(1)) / 10 if compliance_match else 0.5
        except:
            compliance = 0.5
        
        # Prepare results
        results = {
            "evaluation_time": datetime.datetime.now().isoformat(),
            "completeness": completeness,
            "legal_compliance": compliance,
            "semantic_similarity": semantic_similarity,
            "completeness_evaluation": completeness_evaluation,
            "compliance_evaluation": compliance_evaluation
        }
        
        # Generate unique ID for the evaluation
        eval_id = str(uuid.uuid4())
        
        # Save the evaluation results
        results_path = self.results_dir / f"draft_eval_{eval_id}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update metrics history
        self._update_metrics("drafting", {
            "completeness": completeness,
            "legal_compliance": compliance,
            "semantic_similarity": semantic_similarity
        })
        
        return results
    
    def evaluate_translation(self, original_text: str, translated_text: str, 
                            target_language: str, ground_truth_translation: str = None) -> Dict:
        """Evaluate a legal translation against a ground truth or using LLM."""
        # If no ground truth provided, use LLM for evaluation
        if not ground_truth_translation:
            template = f"""
            You are a legal expert fluent in both English and {target_language}.
            
            You are tasked with evaluating the quality of a legal translation from English to {target_language}.
            
            Original English text:
            {original_text}
            
            Translation to {target_language}:
            {translated_text}
            
            Please evaluate the translation on:
            1. Accuracy (correct transfer of information)
            2. Legal terminology correctness
            3. Cultural and jurisdictional appropriateness
            
            Rate each category on a scale of 0 to 10, where 10 is perfect.
            Provide a brief explanation for your ratings.
            
            Output in the format:
            "Accuracy Rating: X/10"
            "Terminology Rating: Y/10"
            "Cultural Appropriateness Rating: Z/10"
            Followed by your explanation.
            """
            
            prompt = PromptTemplate(template=template, input_variables=[])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            evaluation = chain.run({})
            
            # Extract ratings
            try:
                accuracy_match = re.search(r"Accuracy Rating:\s*(\d+(?:\.\d+)?)/10", evaluation)
                accuracy = float(accuracy_match.group(1)) / 10 if accuracy_match else 0.5
                
                terminology_match = re.search(r"Terminology Rating:\s*(\d+(?:\.\d+)?)/10", evaluation)
                terminology = float(terminology_match.group(1)) / 10 if terminology_match else 0.5
                
                cultural_match = re.search(r"Cultural Appropriateness Rating:\s*(\d+(?:\.\d+)?)/10", evaluation)
                cultural = float(cultural_match.group(1)) / 10 if cultural_match else 0.5
            except:
                accuracy = terminology = cultural = 0.5
                
            # Prepare results
            results = {
                "evaluation_time": datetime.datetime.now().isoformat(),
                "target_language": target_language,
                "accuracy": accuracy,
                "terminology_correctness": terminology,
                "cultural_appropriateness": cultural,
                "evaluation": evaluation
            }
        else:
            # Compare with ground truth using embeddings
            try:
                truth_embedding = self.embeddings.embed_query(ground_truth_translation)
                translation_embedding = self.embeddings.embed_query(translated_text)
                
                # Convert to numpy arrays and reshape for cosine_similarity
                truth_embedding_np = np.array(truth_embedding).reshape(1, -1)
                translation_embedding_np = np.array(translation_embedding).reshape(1, -1)
                
                semantic_similarity = float(cosine_similarity(truth_embedding_np, translation_embedding_np)[0][0])
                
                # Use semantic similarity as a proxy for accuracy
                accuracy = semantic_similarity
            except Exception as e:
                accuracy = 0.5
                
            # Use LLM to evaluate terminology and cultural appropriateness
            template = f"""
            You are a legal expert fluent in both English and {target_language}.
            
            You are tasked with evaluating the quality of a legal translation from English to {target_language}.
            
            Original English text:
            {original_text}
            
            Translation to {target_language}:
            {translated_text}
            
            Ground truth professional translation:
            {ground_truth_translation}
            
            Please evaluate the AI translation compared to the ground truth on:
            1. Legal terminology correctness
            2. Cultural and jurisdictional appropriateness
            
            Rate each category on a scale of 0 to 10, where 10 is perfect.
            Provide a brief explanation for your ratings.
            
            Output in the format:
            "Terminology Rating: Y/10"
            "Cultural Appropriateness Rating: Z/10"
            Followed by your explanation.
            """
            
            prompt = PromptTemplate(template=template, input_variables=[])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            evaluation = chain.run({})
            
            # Extract ratings
            try:
                terminology_match = re.search(r"Terminology Rating:\s*(\d+(?:\.\d+)?)/10", evaluation)
                terminology = float(terminology_match.group(1)) / 10 if terminology_match else 0.5
                
                cultural_match = re.search(r"Cultural Appropriateness Rating:\s*(\d+(?:\.\d+)?)/10", evaluation)
                cultural = float(cultural_match.group(1)) / 10 if cultural_match else 0.5
            except:
                terminology = cultural = 0.5
                
            # Prepare results
            results = {
                "evaluation_time": datetime.datetime.now().isoformat(),
                "target_language": target_language,
                "accuracy": accuracy,
                "terminology_correctness": terminology,
                "cultural_appropriateness": cultural,
                "evaluation": evaluation,
                "has_ground_truth": True
            }
        
        # Generate unique ID for the evaluation
        eval_id = str(uuid.uuid4())
        
        # Save the evaluation results
        results_path = self.results_dir / f"translation_eval_{eval_id}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update metrics history
        self._update_metrics("translation", {
            "accuracy": accuracy,
            "terminology_correctness": terminology,
            "cultural_appropriateness": cultural
        })
        
        return results
    
    def _update_metrics(self, category: str, metrics: Dict):
        """Update metrics history."""
        metrics_path = self.results_dir / "metrics_history.json"
        
        try:
            with open(metrics_path, 'r') as f:
                metrics_history = json.load(f)
                
            # Add timestamp to metrics
            metrics_with_time = {
                **metrics, 
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Update each metric
            for metric_name, value in metrics.items():
                if metric_name in metrics_history[category]:
                    metrics_history[category][metric_name].append({
                        "value": value,
                        "timestamp": metrics_with_time["timestamp"]
                    })
            
            # Save updated metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics_history, f, indent=2)
        except Exception as e:
            # If there's an error, reinitialize metrics tracking
            self._init_metrics_tracking()
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all evaluation metrics."""
        metrics_path = self.results_dir / "metrics_history.json"
        
        if not metrics_path.exists():
            return {}
            
        try:
            with open(metrics_path, 'r') as f:
                metrics_history = json.load(f)
                
            summary = {}
            for category, metrics in metrics_history.items():
                summary[category] = {}
                for metric_name, values in metrics.items():
                    if values:
                        recent_values = [item["value"] for item in values[-10:]]
                        summary[category][metric_name] = {
                            "current": values[-1]["value"] if values else 0,
                            "avg_recent": sum(recent_values) / len(recent_values) if recent_values else 0,
                            "trend": "up" if len(values) >= 2 and values[-1]["value"] > values[-2]["value"] else 
                                    "down" if len(values) >= 2 and values[-1]["value"] < values[-2]["value"] else "stable"
                        }
            
            return summary
        except Exception as e:
            return {}