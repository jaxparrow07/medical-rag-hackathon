"""
Medical Named Entity Recognition (NER) Module
Uses scispacy for extracting medical entities from text
"""

import logging
from typing import Dict, List, Set
import re
import json

logger = logging.getLogger(__name__)


class MedicalNER:
    """
    Medical Named Entity Recognition using scispacy

    Extracts:
    - Diseases/Conditions
    - Drugs/Chemicals
    - Procedures/Treatments
    - Anatomical terms
    - Medical abbreviations
    """

    def __init__(self, model_name: str = "en_core_sci_md"):
        """
        Initialize Medical NER

        Args:
            model_name: scispacy model to use
                - en_core_sci_sm: Small model (fast, less accurate)
                - en_core_sci_md: Medium model (balanced) - DEFAULT
                - en_core_sci_lg: Large model (slow, more accurate)
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load scispacy model with fallback"""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"✅ Loaded scispacy model: {self.model_name}")

        except OSError:
            logger.warning(f"⚠️  Model {self.model_name} not found. Attempting to download...")
            try:
                import subprocess
                subprocess.run(
                    ["pip", "install", f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{self.model_name}-0.5.4.tar.gz"],
                    check=True
                )
                import spacy
                self.nlp = spacy.load(self.model_name)
                logger.info(f"✅ Downloaded and loaded: {self.model_name}")

            except Exception as e:
                logger.warning(f"Failed to download {self.model_name}: {e}")
                logger.info("Falling back to en_core_web_sm (general English model)")

                try:
                    import spacy
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    import spacy
                    self.nlp = spacy.load("en_core_web_sm")

        except Exception as e:
            logger.error(f"❌ Failed to load any spacy model: {e}")
            self.nlp = None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text

        Returns:
            {
                'diseases': [...],
                'drugs': [...],
                'procedures': [...],
                'anatomy': [...],
                'all_entities': [...]
            }
        """
        if not self.nlp:
            return {
                'diseases': [],
                'drugs': [],
                'procedures': [],
                'anatomy': [],
                'all_entities': []
            }

        # Process text with spacy
        doc = self.nlp(text)

        # Extract entities by type
        diseases = set()
        drugs = set()
        procedures = set()
        anatomy = set()
        all_entities = set()

        for ent in doc.ents:
            entity_text = ent.text.strip()
            entity_label = ent.label_

            all_entities.add(entity_text)

            # Map entity labels to categories
            # scispacy uses different label sets depending on model
            if entity_label in ['DISEASE', 'DISORDER', 'PHENOTYPE', 'PROBLEM']:
                diseases.add(entity_text)
            elif entity_label in ['CHEMICAL', 'DRUG', 'MEDICATION']:
                drugs.add(entity_text)
            elif entity_label in ['PROCEDURE', 'TREATMENT', 'TEST']:
                procedures.add(entity_text)
            elif entity_label in ['ANATOMY', 'ORGAN', 'CELL', 'TISSUE']:
                anatomy.add(entity_text)

        # Also use rule-based extraction for common medical terms
        rule_based = self._rule_based_extraction(text)

        return {
            'diseases': sorted(list(diseases.union(rule_based['diseases']))),
            'drugs': sorted(list(drugs.union(rule_based['drugs']))),
            'procedures': sorted(list(procedures.union(rule_based['procedures']))),
            'anatomy': sorted(list(anatomy.union(rule_based['anatomy']))),
            'all_entities': sorted(list(all_entities))
        }

    def _rule_based_extraction(self, text: str) -> Dict[str, Set[str]]:
        """
        Rule-based extraction for common medical terms
        Fallback when NER doesn't catch everything
        """
        diseases = set()
        drugs = set()
        procedures = set()
        anatomy = set()

        # Common disease suffixes
        disease_patterns = [
            r'\b\w+itis\b',  # inflammation (e.g., arthritis, hepatitis)
            r'\b\w+oma\b',   # tumor (e.g., carcinoma, melanoma)
            r'\b\w+osis\b',  # condition (e.g., cirrhosis, psychosis)
            r'\b\w+pathy\b', # disease (e.g., neuropathy, myopathy)
        ]

        for pattern in disease_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            diseases.update([m.lower() for m in matches])

        # Common drug suffixes
        drug_patterns = [
            r'\b\w+cillin\b',  # antibiotics (e.g., penicillin, amoxicillin)
            r'\b\w+statin\b',  # cholesterol drugs (e.g., atorvastatin)
            r'\b\w+prazole\b', # PPIs (e.g., omeprazole)
            r'\b\w+olol\b',    # beta blockers (e.g., metoprolol)
        ]

        for pattern in drug_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            drugs.update([m.lower() for m in matches])

        # Common procedure suffixes
        procedure_patterns = [
            r'\b\w+ectomy\b',  # removal (e.g., appendectomy)
            r'\b\w+oscopy\b',  # viewing (e.g., colonoscopy)
            r'\b\w+plasty\b',  # surgical repair (e.g., rhinoplasty)
            r'\b\w+otomy\b',   # cutting (e.g., laparotomy)
        ]

        for pattern in procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            procedures.update([m.lower() for m in matches])

        return {
            'diseases': diseases,
            'drugs': drugs,
            'procedures': procedures,
            'anatomy': anatomy
        }

    def extract_medical_abbreviations(self, text: str) -> Dict[str, str]:
        """
        Extract medical abbreviations from text

        Pattern: "Full Term (ABBR)" or common medical abbreviations

        Returns: {abbreviation: full_term or 'unknown'}
        """
        abbreviations = {}

        # Pattern: Full Term (ABBR)
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+\(([A-Z]{2,})\)'
        matches = re.findall(pattern, text)

        for full_term, abbr in matches:
            abbreviations[abbr] = full_term.strip()

        # Common medical abbreviations (add more as needed)
        common_abbrs = {
            'MI': 'Myocardial Infarction',
            'COPD': 'Chronic Obstructive Pulmonary Disease',
            'HTN': 'Hypertension',
            'DM': 'Diabetes Mellitus',
            'CHF': 'Congestive Heart Failure',
            'CVA': 'Cerebrovascular Accident',
            'PE': 'Pulmonary Embolism',
            'DVT': 'Deep Vein Thrombosis',
            'UTI': 'Urinary Tract Infection',
            'CAD': 'Coronary Artery Disease',
            'ACE': 'Angiotensin-Converting Enzyme',
            'ARB': 'Angiotensin Receptor Blocker',
            'NSAID': 'Non-Steroidal Anti-Inflammatory Drug',
            'CBC': 'Complete Blood Count',
            'CXR': 'Chest X-Ray',
            'CT': 'Computed Tomography',
            'MRI': 'Magnetic Resonance Imaging',
        }

        # Find common abbreviations in text
        for abbr, full_term in common_abbrs.items():
            if re.search(r'\b' + abbr + r'\b', text):
                if abbr not in abbreviations:
                    abbreviations[abbr] = full_term

        return abbreviations

    def enrich_chunk_metadata(self, chunk_text: str, existing_metadata: Dict = None) -> Dict:
        """
        Enrich chunk with medical entity metadata

        Args:
            chunk_text: The text chunk
            existing_metadata: Existing metadata to merge with

        Returns:
            Metadata with medical entities (flattened for ChromaDB compatibility)
        """
        if existing_metadata is None:
            existing_metadata = {}

        # Extract entities
        entities = self.extract_entities(chunk_text)

        # Extract abbreviations
        abbreviations = self.extract_medical_abbreviations(chunk_text)

        # Flatten nested structures for ChromaDB compatibility
        # ChromaDB only accepts: str, int, float, bool, or None as metadata values
        enriched_metadata = {
            **existing_metadata,
            # Serialize nested structures as JSON strings
            'medical_entities_json': json.dumps(entities),
            'abbreviations_json': json.dumps(abbreviations),
            # Keep individual entity lists as comma-separated strings for easier filtering
            'diseases': ', '.join(entities['diseases']) if entities['diseases'] else '',
            'drugs': ', '.join(entities['drugs']) if entities['drugs'] else '',
            'procedures': ', '.join(entities['procedures']) if entities['procedures'] else '',
            'anatomy': ', '.join(entities['anatomy']) if entities['anatomy'] else '',
            'all_entities': ', '.join(entities['all_entities']) if entities['all_entities'] else '',
            # Add counts and flags
            'entity_count': len(entities['all_entities']),
            'has_diseases': len(entities['diseases']) > 0,
            'has_drugs': len(entities['drugs']) > 0,
            'has_procedures': len(entities['procedures']) > 0,
        }

        return enriched_metadata

    def get_entity_summary(self, chunks: List[Dict]) -> Dict:
        """
        Get summary of all entities across multiple chunks

        Args:
            chunks: List of chunk dicts with 'text' key

        Returns:
            Summary of all unique entities found
        """
        all_diseases = set()
        all_drugs = set()
        all_procedures = set()
        all_anatomy = set()
        all_abbreviations = {}

        for chunk in chunks:
            text = chunk.get('text', '')
            entities = self.extract_entities(text)
            abbreviations = self.extract_medical_abbreviations(text)

            all_diseases.update(entities['diseases'])
            all_drugs.update(entities['drugs'])
            all_procedures.update(entities['procedures'])
            all_anatomy.update(entities['anatomy'])
            all_abbreviations.update(abbreviations)

        return {
            'total_unique_diseases': len(all_diseases),
            'total_unique_drugs': len(all_drugs),
            'total_unique_procedures': len(all_procedures),
            'total_unique_anatomy': len(all_anatomy),
            'total_abbreviations': len(all_abbreviations),
            'diseases': sorted(list(all_diseases)),
            'drugs': sorted(list(all_drugs)),
            'procedures': sorted(list(all_procedures)),
            'anatomy': sorted(list(all_anatomy)),
            'abbreviations': all_abbreviations
        }


def test_medical_ner():
    """Test the medical NER module"""
    ner = MedicalNER()

    # Test text
    test_text = """
    The patient presented with acute myocardial infarction (MI) and was treated with aspirin
    and atorvastatin. A coronary angiography was performed, revealing significant coronary
    artery disease (CAD). The patient has a history of hypertension (HTN) and diabetes mellitus (DM).
    Previous medications include metformin and lisinopril. The patient underwent percutaneous
    coronary intervention (PCI) and was started on clopidogrel for dual antiplatelet therapy.
    """

    print("\n" + "="*60)
    print("Testing Medical NER")
    print("="*60 + "\n")

    # Extract entities
    entities = ner.extract_entities(test_text)

    print("Extracted Entities:")
    print(f"\nDiseases: {entities['diseases']}")
    print(f"\nDrugs: {entities['drugs']}")
    print(f"\nProcedures: {entities['procedures']}")

    # Extract abbreviations
    abbreviations = ner.extract_medical_abbreviations(test_text)

    print(f"\nAbbreviations:")
    for abbr, full_term in abbreviations.items():
        print(f"  {abbr}: {full_term}")

    # Test metadata enrichment
    print("\n" + "="*60)
    print("Testing Metadata Enrichment")
    print("="*60 + "\n")

    metadata = ner.enrich_chunk_metadata(
        test_text,
        existing_metadata={'source': 'test.pdf', 'page': 1}
    )

    print(f"Entity Count: {metadata['entity_count']}")
    print(f"Has Diseases: {metadata['has_diseases']}")
    print(f"Has Drugs: {metadata['has_drugs']}")
    print(f"Has Procedures: {metadata['has_procedures']}")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    test_medical_ner()
