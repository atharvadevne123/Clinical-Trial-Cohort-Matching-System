"""Step 6: FHIR Client for EHR Integration"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FHIRClient:
    def __init__(self, base_url: str = "http://fhir-server:8080/fhir"):
        self.base_url = base_url
        logger.info(f"🔗 FHIR client initialized: {base_url}")
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return {
            "resourceType": "Patient",
            "id": patient_id,
            "name": [{"given": ["John"], "family": "Doe"}],
            "birthDate": "1960-01-15",
            "gender": "male",
            "telecom": [{"system": "email", "value": "john@example.com"}],
            "address": [{"postalCode": "12345"}]
        }
    
    def get_patient_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "resourceType": "Condition",
                "id": "cond_1",
                "code": {
                    "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I48.91", "display": "Atrial Fibrillation"}],
                    "text": "Atrial Fibrillation"
                }
            }
        ]
    
    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "resourceType": "MedicationStatement",
                "id": "med_1",
                "medicationCodeableConcept": {
                    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "B01AC04", "display": "Warfarin"}],
                    "text": "Warfarin"
                },
                "status": "active"
            }
        ]
    
    def parse_patient(self, fhir_patient: Dict[str, Any]) -> Dict[str, Any]:
        name = fhir_patient.get("name", [{}])[0]
        return {
            "fhir_id": fhir_patient.get("id"),
            "first_name": name.get("given", [""])[0] if name.get("given") else "",
            "last_name": name.get("family", ""),
            "date_of_birth": fhir_patient.get("birthDate"),
            "gender": fhir_patient.get("gender", "unknown"),
            "postal_code": fhir_patient.get("address", [{}])[0].get("postalCode")
        }
    
    def parse_condition(self, fhir_condition: Dict[str, Any]) -> Dict[str, Any]:
        coding = fhir_condition.get("code", {}).get("coding", [{}])[0]
        return {
            "icd10_code": coding.get("code"),
            "display": fhir_condition.get("code", {}).get("text", coding.get("display", ""))
        }
    
    def parse_medication(self, fhir_medication: Dict[str, Any]) -> Dict[str, Any]:
        med_ref = fhir_medication.get("medicationCodeableConcept", {})
        coding = med_ref.get("coding", [{}])[0]
        return {
            "medication_code": coding.get("code"),
            "display": med_ref.get("text", coding.get("display", ""))
        }
    
    def fetch_complete_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        fhir_patient = self.get_patient(patient_id)
        patient_data = self.parse_patient(fhir_patient)
        
        fhir_conditions = self.get_patient_conditions(patient_id)
        conditions = [self.parse_condition(cond) for cond in fhir_conditions]
        patient_data["conditions"] = conditions
        
        fhir_medications = self.get_patient_medications(patient_id)
        medications = [self.parse_medication(med) for med in fhir_medications]
        patient_data["medications"] = medications
        
        return patient_data

fhir_client = FHIRClient()
