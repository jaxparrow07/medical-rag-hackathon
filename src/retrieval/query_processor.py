from rapidfuzz import fuzz, process
import re

class QueryProcessor:
    def __init__(self, medical_terms_file: str = None):
        """Initialize with medical vocabulary"""
        self.medical_terms = self.load_medical_terms(medical_terms_file)
    
    def load_medical_terms(self, filepath: str = None) -> set:
        """Load medical terms for spell correction"""
        # Basic medical terms (expand this list)
        terms = {
            'diabetes', 'hypertension', 'pneumonia', 'cardiovascular',
            'thrombosis', 'anemia', 'leukemia', 'nephrology', 'cardiology',
            'neurology', 'oncology', 'hematology', 'endocrine', 'gastric',
            'hepatic', 'renal', 'pulmonary', 'dermatology', 'orthopedic'
        }
        
        if filepath:
            with open(filepath) as f:
                terms.update(line.strip().lower() for line in f)
        
        return terms
    
    def correct_query(self, query: str, threshold: int = 80) -> str:
        """Correct typos in query using fuzzy matching"""
        words = query.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if word needs correction (not in medical terms)
            if len(word) > 3 and word_lower not in self.medical_terms:
                # Find best match from medical terms
                match = process.extractOne(
                    word_lower,
                    self.medical_terms,
                    scorer=fuzz.ratio,
                    score_cutoff=threshold
                )
                
                if match:
                    corrected_words.append(match[0])
                    print(f"Corrected: {word} → {match[0]}")
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def expand_query(self, query: str) -> str:
        """Add medical synonyms/expansions"""
        # Add common medical abbreviations
        expansions = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'mi': 'myocardial infarction',
            'cva': 'cerebrovascular accident stroke',
            'copd': 'chronic obstructive pulmonary disease',
            'uti': 'urinary tract infection'
        }
        
        query_lower = query.lower()
        for abbr, expansion in expansions.items():
            if abbr in query_lower:
                query += f" {expansion}"
        
        return query
