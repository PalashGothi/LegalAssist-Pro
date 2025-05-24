
import matplotlib.pyplot as plt
import pandas as pd

def compare_summarization_techniques(document, ai_summary, traditional_summary, hybrid_summary):
    """Compare different summarization techniques."""
    try:
        # Placeholder metrics
        metrics = {
            "AI Summary Length": len(ai_summary.split()) if ai_summary else 0,
            "Traditional Summary Length": len(traditional_summary.split()) if traditional_summary else 0,
            "Hybrid Summary Length": len(hybrid_summary.split()) if hybrid_summary else 0
        }
        
        # Qualitative analysis
        qualitative_analysis = "AI summary is concise, traditional summary is detailed, hybrid summary balances both."
        
        # Create a simple visualization
        plt.figure(figsize=(8, 4))
        lengths = [metrics["AI Summary Length"], metrics["Traditional Summary Length"], metrics["Hybrid Summary Length"]]
        labels = ["AI", "Traditional", "Hybrid"]
        plt.bar(labels, lengths)
        plt.title("Summary Length Comparison")
        plt.ylabel("Word Count")
        
        return {
            "metrics": metrics,
            "qualitative_analysis": qualitative_analysis,
            "visualization": plt
        }
    except Exception as e:
        raise Exception(f"Error comparing summarization techniques: {str(e)}")

def compare_human_ai_documents(human_text, ai_text):
    """Compare human-generated and AI-generated documents."""
    try:
        # Placeholder metrics
        metrics = {
            "Human Text Length": len(human_text.split()),
            "AI Text Length": len(ai_text.split()),
            "Overlap Words": len(set(human_text.split()) & set(ai_text.split()))
        }
        
        # Analysis
        analysis = "Human text is detailed and context-rich, while AI text is structured but may lack nuance."
        
        # Differences
        differences = pd.DataFrame({
            "Metric": ["Length Difference", "Unique Words (Human)", "Unique Words (AI)"],
            "Value": [
                abs(metrics["Human Text Length"] - metrics["AI Text Length"]),
                len(set(human_text.split()) - set(ai_text.split())),
                len(set(ai_text.split()) - set(human_text.split()))
            ]
        })
        
        # Visualization
        plt.figure(figsize=(8, 4))
        plt.bar(["Human", "AI"], [metrics["Human Text Length"], metrics["AI Text Length"]])
        plt.title("Document Length Comparison")
        plt.ylabel("Word Count")
        
        return {
            "metrics": metrics,
            "analysis": analysis,
            "differences": differences,
            "visualization": plt
        }
    except Exception as e:
        raise Exception(f"Error comparing human and AI documents: {str(e)}")


