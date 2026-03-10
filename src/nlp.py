"""Step 5: NLP Pipeline for Clinical Entity Extraction"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ClinicalNLPProcessor:
    """Simple NLP processor using keyword matching"""
    
    def __init__(self):
        self.condition_keywords = {
            "atrial fibrillation": {"codes": ["I48.91"], "type": "CONDITION"},
            "afib": {"codes": ["I48.91"], "type": "CONDITION"},
            "hypertension": {"codes": ["I10"], "type": "CONDITION"},
            "high blood pressure": {"codes": ["I10"], "type": "CONDITION"},
            "diabetes": {"codes": ["E11"], "type": "CONDITION"},
            "type 2 diabetes": {"codes": ["E11"], "type": "CONDITION"},
            "heart failure": {"codes": ["I50"], "type": "CONDITION"},
            "chronic kidney disease": {"codes": ["N18"], "type": "CONDITION"},
            "ckd": {"codes": ["N18"], "type": "CONDITION"},
        }
        
        self.medication_keywords = {
            "warfarin": {"codes": ["B01AC04"], "type": "MEDICATION"},
            "aspirin": {"codes": ["N02BA01"], "type": "MEDICATION"},
            "metformin": {"codes": ["A10BA02"], "type": "MEDICATION"},
            "lisinopril": {"codes": ["C09AA01"], "type": "MEDICATION"},
            "atorvastatin": {"codes": ["C10AA05"], "type": "MEDICATION"},
        }
        
        self.severity_markers = {
            "severe": 3,
            "critical": 4,
            "moderate": 2,
            "mild": 1,
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract clinical entities from text"""
        if not text:
            return {"conditions": [], "medications": [], "symptoms": [], "severity": {}}
        
        text_lower = text.lower()
        entities = {
            "conditions": [],
            "medications": [],
            "symptoms": [],
            "severity": {}
        }
        
        for keyword, info in self.condition_keywords.items():
            if keyword in text_lower:
                entity = {
                    "text": keyword,
                    "codes": info["codes"],
                    "type": info["type"],
                    "confidence": 0.8
                }
                entities["conditions"].append(entity)
                
                severity = self._extract_severity(text_lower, keyword)
                if severity:
                    entities["severity"][keyword] = severity
        
        for keyword, info in self.medication_keywords.items():
            if keyword in text_lower:
                entity = {
                    "text": keyword,
                    "codes": info["codes"],
                    "type": info["type"],
                    "confidence": 0.8
                }
                entities["medications"].append(entity)
        
        symptoms = self._extract_symptoms(text_lower)
        entities["symptoms"] = symptoms
        
        return entities
    
    def _extract_severity(self, text: str, entity: str) -> str:
        """Extract severity markers near an entity"""
        entity_pos = text.find(entity)
        if entity_pos == -1:
            return None
        
        context = text[max(0, entity_pos - 100):entity_pos + 100]
        
        for severity_word in self.severity_markers.keys():
            if severity_word in context:
                return severity_word
        
        return None
    
    def _extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Extract common symptoms"""
        symptom_keywords = {
            "chest pain": "chest discomfort",
            "shortness of breath": "dyspnea",
            "difficulty breathing": "dyspnea",
            "fatigue": "tiredness",
            "dizziness": "vertigo",
            "headache": "headache",
            "nausea": "nausea",
            "fever": "elevated temperature",
            "cough": "cough",
        }
        
        symptoms = []
        for keyword in symptom_keywords.keys():
            if keyword in text:
                symptoms.append({
                    "text": keyword,
                    "confidence": 0.7
                })
        
        return symptoms
    
    def summarize_clinical_profile(self, text: str) -> Dict[str, Any]:
        """Create a clinical profile summary"""
        entities = self.extract_entities(text)
        
        profile = {
            "num_conditions": len(entities["conditions"]),
            "num_medications": len(entities["medications"]),
            "num_symptoms": len(entities["symptoms"]),
            "conditions": [e["text"] for e in entities["conditions"]],
            "medications": [e["text"] for e in entities["medications"]],
            "symptoms": [e["text"] for e in entities["symptoms"]],
            "disease_burden": "low" if len(entities["conditions"]) <= 1 else "moderate" if len(entities["conditions"]) <= 3 else "high",
        }
        
        return profile


nlp_processor = ClinicalNLPProcessor()
