
import os
from typing import Dict, List, Optional, Tuple
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
import re
import json
from pathlib import Path

class MultilingualProcessor:
    """Processor for multilingual support with focus on Indian languages."""
    
    def __init__(self, llm=None):
        """Initialize the multilingual processor."""
        self.llm = llm
        self.supported_languages = {
            "English": "en",
            "Hindi": "hi",
            "Bengali": "bn",
            "Telugu": "te",
            "Marathi": "mr",
            "Tamil": "ta",
            "Urdu": "ur",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Punjabi": "pa"
        }
        
        # Load legal terminology dictionary for each language
        self.terminology_dict = self._load_terminology_dicts()
        
    def _load_terminology_dicts(self) -> Dict[str, Dict[str, str]]:
        """Load legal terminology dictionaries for each language."""
        terminology_dir = Path("./terminology")
        terminology_dir.mkdir(exist_ok=True)
        
        terminology = {}
        
        # Sample legal terms
        sample_terms = {
            "agreement": {
                "en": "agreement",
                "hi": "अनुबंध",
                "bn": "চুক্তি",
                "te": "ఒప్పందం",
                "mr": "करार",
                "ta": "ஒப்பந்தம்",
                "ur": "معاہدہ",
                "gu": "કરાર",
                "kn": "ಒಪ್ಪಂದ",
                "ml": "കരാർ",
                "pa": "ਸਮਝੌਤਾ"
            }
        }
        
        # Initialize with sample data for each language
        for lang, code in self.supported_languages.items():
            lang_file = terminology_dir / f"{code}_legal_terms.json"
            
            if not lang_file.exists():
                terms = {}
                for term, translations in sample_terms.items():
                    if code in translations:
                        terms[term] = translations[code]
                
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(terms, f, ensure_ascii=False, indent=2)
            
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    terminology[code] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                terminology[code] = {}
        
        return terminology
    
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language with legal terminology handling."""
        if target_language not in self.supported_languages.values():
            if target_language in self.supported_languages:
                target_language = self.supported_languages[target_language]
            else:
                raise ValueError(f"Unsupported language: {target_language}")
        
        if target_language == "en":
            return text
        
        legal_terms = self.terminology_dict.get(target_language, {})
        marked_text = text
        markers = {}
        
        for i, (term, translation) in enumerate(legal_terms.items()):
            marker = f"__LEGAL_TERM_{i}__"
            markers[marker] = translation
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            marked_text = pattern.sub(marker, marked_text)
        
        max_chars = 4500  # Google Translator limit
        
        if len(marked_text) <= max_chars:
            try:
                translated = GoogleTranslator(source='auto', target=target_language).translate(marked_text)
            except Exception as e:
                return f"Translation error: {str(e)}"
        else:
            chunks = [marked_text[i:i+max_chars] for i in range(0, len(marked_text), max_chars)]
            translated_chunks = []
            
            for chunk in chunks:
                try:
                    translated_chunk = GoogleTranslator(source='auto', target=target_language).translate(chunk)
                    translated_chunks.append(translated_chunk)
                except Exception as e:
                    translated_chunks.append(f"[Translation error: {str(e)}]")
            
            translated = "".join(translated_chunks)
        
        for marker, translation in markers.items():
            translated = translated.replace(marker, translation)
        
        return translated
    
    def translate_document(self, document_text: str, target_language: str) -> str:
        """Translate a legal document preserving structure."""
        lines = document_text.split('\n')
        translated_lines = []
        
        for line in lines:
            if line.strip():
                translated_line = self.translate_text(line, target_language)
                translated_lines.append(translated_line)
            else:
                translated_lines.append('')
        
        return '\n'.join(translated_lines)
    
    def process_text(self, text: str, source_language: str, target_language: str, options: List[str]) -> Dict:
        """Process text with translation and other options."""
        try:
            # Translate using existing method
            translated_text = self.translate_document(text, target_language)
            
            # Handle options
            preserved_terms = []
            accuracy_report = "Translation completed."
            
            if "Legal Terms Preservation" in options:
                legal_terms = self.terminology_dict.get(self.supported_languages.get(target_language, target_language), {})
                preserved_terms = [
                    {"original": term, "translated": translation}
                    for term, translation in legal_terms.items()
                    if term in text
                ]
            
            if "Format Preservation" in options:
                # Already handled by translate_document
                accuracy_report += " Format preserved."
            
            if "Cultural Adaptation" in options and self.llm:
                template = f"""
                Adapt the following translated text to cultural norms of {target_language}:
                
                {translated_text}
                
                Provide the culturally adapted version.
                """
                prompt = PromptTemplate(template=template, input_variables=[])
                chain = LLMChain(llm=self.llm, prompt=prompt)
                translated_text = chain.run({})
                accuracy_report += " Cultural adaptation applied."
            
            return {
                "translated_text": translated_text,
                "preserved_terms": preserved_terms,
                "accuracy_report": accuracy_report
            }
        except Exception as e:
            raise Exception(f"Error processing text: {str(e)}")
    
    def verify_translation_quality(self, original_text: str, translated_text: str, 
                                target_language: str) -> Dict[str, any]:
        """Verify translation quality using LLM."""
        try:
            back_translation = GoogleTranslator(
                source=self.supported_languages.get(target_language, target_language), 
                target='en'
            ).translate(translated_text[:5000])  # Limit to prevent excessive tokens
        except Exception as e:
            back_translation = f"Back-translation error: {str(e)}"
        
        template = f"""
        Evaluate this translation from English to {target_language}:
        
        Original:
        {original_text[:2000]}
        
        Back-translation:
        {back_translation}
        
        Analyze:
        1. Semantic accuracy (1-5)
        2. Legal terminology (1-5)
        3. Fluency (1-5)
        Provide specific feedback.
        """
        
        prompt = PromptTemplate(template=template, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        evaluation = chain.run({})
        
        return {
            "original": original_text,
            "translated": translated_text,
            "back_translation": back_translation,
            "evaluation": evaluation
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of provided text."""
        try:
            detected = GoogleTranslator(source='auto', target='en').detect_language(text[:100])
            for name, code in self.supported_languages.items():
                if code == detected:
                    return name
            return "Unknown"
        except:
            return "Unknown"
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return list(self.supported_languages.keys())
    
    def add_terminology(self, language_code: str, term: str, translation: str) -> bool:
        """Add new term to terminology dictionary."""
        if language_code not in self.supported_languages.values():
            if language_code in self.supported_languages:
                language_code = self.supported_languages[language_code]
            else:
                return False
                
        if language_code not in self.terminology_dict:
            self.terminology_dict[language_code] = {}
            
        self.terminology_dict[language_code][term] = translation
        
        try:
            terminology_dir = Path("./terminology")
            terminology_dir.mkdir(exist_ok=True)
            
            lang_file = terminology_dir / f"{language_code}_legal_terms.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(self.terminology_dict[language_code], f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False