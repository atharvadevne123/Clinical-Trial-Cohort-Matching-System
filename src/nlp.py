"""NLP Pipeline for Clinical Entity Extraction"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Words that negate the following entity (checked in a 60-char window before the keyword)
_NEGATION_WORDS = [
    "no ", "not ", "without ", "denies ", "denied ",
    "no history of ", "negative for ", "rules out ",
    "ruled out ", "absence of ", "never had ",
]


class ClinicalNLPProcessor:
    """Keyword-based clinical NLP with negation detection."""

    def __init__(self):
        self.condition_keywords = {
            "atrial fibrillation": {"codes": ["I48.91"], "type": "CONDITION"},
            "afib": {"codes": ["I48.91"], "type": "CONDITION"},
            "hypertension": {"codes": ["I10"], "type": "CONDITION"},
            "high blood pressure": {"codes": ["I10"], "type": "CONDITION"},
            "diabetes mellitus": {"codes": ["E11"], "type": "CONDITION"},
            "type 2 diabetes": {"codes": ["E11"], "type": "CONDITION"},
            "diabetes": {"codes": ["E11"], "type": "CONDITION"},
            "heart failure": {"codes": ["I50"], "type": "CONDITION"},
            "chronic kidney disease": {"codes": ["N18"], "type": "CONDITION"},
            "ckd": {"codes": ["N18"], "type": "CONDITION"},
            "copd": {"codes": ["J44"], "type": "CONDITION"},
            "chronic obstructive pulmonary disease": {"codes": ["J44"], "type": "CONDITION"},
            "asthma": {"codes": ["J45"], "type": "CONDITION"},
            "depression": {"codes": ["F32"], "type": "CONDITION"},
            "major depressive disorder": {"codes": ["F32"], "type": "CONDITION"},
            "obesity": {"codes": ["E66"], "type": "CONDITION"},
            "stroke": {"codes": ["I63"], "type": "CONDITION"},
            "ischemic stroke": {"codes": ["I63"], "type": "CONDITION"},
            "osteoporosis": {"codes": ["M81"], "type": "CONDITION"},
            "rheumatoid arthritis": {"codes": ["M05"], "type": "CONDITION"},
            "migraine": {"codes": ["G43"], "type": "CONDITION"},
            "alzheimer": {"codes": ["G30"], "type": "CONDITION"},
            "cancer": {"codes": ["C80"], "type": "CONDITION"},
            "breast cancer": {"codes": ["C50"], "type": "CONDITION"},
            "lung cancer": {"codes": ["C34"], "type": "CONDITION"},
            "prostate cancer": {"codes": ["C61"], "type": "CONDITION"},
        }

        self.medication_keywords = {
            "warfarin": {"codes": ["B01AA03"], "type": "MEDICATION"},
            "apixaban": {"codes": ["B01AF02"], "type": "MEDICATION"},
            "aspirin": {"codes": ["N02BA01"], "type": "MEDICATION"},
            "metformin": {"codes": ["A10BA02"], "type": "MEDICATION"},
            "insulin": {"codes": ["A10AB01"], "type": "MEDICATION"},
            "lisinopril": {"codes": ["C09AA01"], "type": "MEDICATION"},
            "amlodipine": {"codes": ["C08CA01"], "type": "MEDICATION"},
            "atorvastatin": {"codes": ["C10AA05"], "type": "MEDICATION"},
            "metoprolol": {"codes": ["C07AB02"], "type": "MEDICATION"},
            "furosemide": {"codes": ["C03CA01"], "type": "MEDICATION"},
            "sertraline": {"codes": ["N06AB06"], "type": "MEDICATION"},
            "omeprazole": {"codes": ["A02BC01"], "type": "MEDICATION"},
            "albuterol": {"codes": ["R03AC02"], "type": "MEDICATION"},
            "prednisone": {"codes": ["H02AB07"], "type": "MEDICATION"},
        }

        self.severity_markers = {
            "severe": 3,
            "critical": 4,
            "moderate": 2,
            "mild": 1,
            "poorly controlled": 3,
            "well controlled": 1,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract clinical entities from free text, skipping negated mentions."""
        if not text:
            return {"conditions": [], "medications": [], "symptoms": [], "severity": {}}

        text_lower = text.lower()
        entities: Dict[str, Any] = {
            "conditions": [],
            "medications": [],
            "symptoms": [],
            "severity": {},
        }

        for keyword, info in self.condition_keywords.items():
            if keyword in text_lower and not self._is_negated(text_lower, keyword):
                entities["conditions"].append({
                    "text": keyword,
                    "codes": info["codes"],
                    "type": info["type"],
                    "confidence": 0.85,
                })
                severity = self._extract_severity(text_lower, keyword)
                if severity:
                    entities["severity"][keyword] = severity

        for keyword, info in self.medication_keywords.items():
            if keyword in text_lower and not self._is_negated(text_lower, keyword):
                entities["medications"].append({
                    "text": keyword,
                    "codes": info["codes"],
                    "type": info["type"],
                    "confidence": 0.85,
                })

        entities["symptoms"] = self._extract_symptoms(text_lower)
        return entities

    def summarize_clinical_profile(self, text: str) -> Dict[str, Any]:
        """Return a structured summary of the clinical profile in the text."""
        entities = self.extract_entities(text)
        burden = (
            "low" if len(entities["conditions"]) <= 1
            else "moderate" if len(entities["conditions"]) <= 3
            else "high"
        )
        return {
            "num_conditions": len(entities["conditions"]),
            "num_medications": len(entities["medications"]),
            "num_symptoms": len(entities["symptoms"]),
            "conditions": [e["text"] for e in entities["conditions"]],
            "medications": [e["text"] for e in entities["medications"]],
            "symptoms": [e["text"] for e in entities["symptoms"]],
            "severity": entities["severity"],
            "disease_burden": burden,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_negated(self, text: str, keyword: str) -> bool:
        """Return True if the keyword appears immediately after a negation phrase."""
        pos = text.find(keyword)
        if pos == -1:
            return False
        window = text[max(0, pos - 60): pos]
        return any(neg in window for neg in _NEGATION_WORDS)

    def _extract_severity(self, text: str, entity: str) -> Optional[str]:
        pos = text.find(entity)
        if pos == -1:
            return None
        context = text[max(0, pos - 100): pos + len(entity) + 100]
        for word in self.severity_markers:
            if word in context:
                return word
        return None

    def _extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        symptom_map = {
            "chest pain": "chest discomfort",
            "shortness of breath": "dyspnea",
            "difficulty breathing": "dyspnea",
            "fatigue": "fatigue",
            "dizziness": "vertigo",
            "headache": "headache",
            "nausea": "nausea",
            "fever": "elevated temperature",
            "cough": "cough",
            "palpitations": "palpitations",
            "swelling": "oedema",
            "weight gain": "weight gain",
            "weight loss": "weight loss",
        }
        return [
            {"text": kw, "canonical": canonical, "confidence": 0.75}
            for kw, canonical in symptom_map.items()
            if kw in text and not self._is_negated(text, kw)
        ]


nlp_processor = ClinicalNLPProcessor()
