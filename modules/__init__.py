# modules/__init__.py
# Empty file to make 'modules' a Python package
# modules/__init__.py
# modules/__init__.py
from modules.multilingual_processor import MultilingualProcessor
from .legal_summarizer import LegalSummarizer
# Add other exports as needed

__all__ = ['MultilingualProcessor', 'LegalSummarizer']  # Add all your classes