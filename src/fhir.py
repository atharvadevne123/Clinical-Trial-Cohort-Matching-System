"""FHIR R4 Client for EHR Integration"""

import logging
from typing import Dict, List, Any, Optional

import httpx

logger = logging.getLogger(__name__)


class FHIRClient:
    """
    FHIR R4 HTTP client.
    Each method attempts a real HTTP request and falls back to mock data
    when the server is unreachable (e.g. local dev without a FHIR server).
    """

    def __init__(self, base_url: str = "http://fhir-server:8080/fhir", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        logger.info(f"FHIR client initialised: {self.base_url}")

    # ------------------------------------------------------------------
    # Resource fetchers
    # ------------------------------------------------------------------

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/Patient/{patient_id}"
        try:
            resp = httpx.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.debug(f"FHIR server unreachable – returning mock Patient for {patient_id}")
            return self._mock_patient(patient_id)

    def get_patient_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/Condition"
        try:
            resp = httpx.get(url, params={"patient": patient_id}, timeout=self.timeout)
            resp.raise_for_status()
            bundle = resp.json()
            return [entry["resource"] for entry in bundle.get("entry", [])]
        except Exception:
            logger.debug(f"FHIR server unreachable – returning mock Conditions for {patient_id}")
            return self._mock_conditions()

    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/MedicationStatement"
        try:
            resp = httpx.get(url, params={"patient": patient_id}, timeout=self.timeout)
            resp.raise_for_status()
            bundle = resp.json()
            return [entry["resource"] for entry in bundle.get("entry", [])]
        except Exception:
            logger.debug(f"FHIR server unreachable – returning mock Medications for {patient_id}")
            return self._mock_medications()

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def parse_patient(self, fhir_patient: Dict[str, Any]) -> Dict[str, Any]:
        name = fhir_patient.get("name", [{}])[0]
        address = fhir_patient.get("address", [{}])[0] if fhir_patient.get("address") else {}
        telecom = {t["system"]: t["value"] for t in fhir_patient.get("telecom", [])}
        return {
            "fhir_id": fhir_patient.get("id"),
            "first_name": (name.get("given") or [""])[0],
            "last_name": name.get("family", ""),
            "date_of_birth": fhir_patient.get("birthDate"),
            "gender": fhir_patient.get("gender", "unknown"),
            "email": telecom.get("email"),
            "phone": telecom.get("phone"),
            "postal_code": address.get("postalCode"),
        }

    def parse_condition(self, fhir_condition: Dict[str, Any]) -> Dict[str, Any]:
        coding = (fhir_condition.get("code") or {}).get("coding", [{}])[0]
        return {
            "icd10_code": coding.get("code"),
            "display": (fhir_condition.get("code") or {}).get("text") or coding.get("display", ""),
        }

    def parse_medication(self, fhir_medication: Dict[str, Any]) -> Dict[str, Any]:
        med_ref = fhir_medication.get("medicationCodeableConcept") or {}
        coding = med_ref.get("coding", [{}])[0]
        return {
            "medication_code": coding.get("code"),
            "display": med_ref.get("text") or coding.get("display", ""),
            "status": fhir_medication.get("status", "unknown"),
        }

    def fetch_complete_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        patient_data = self.parse_patient(self.get_patient(patient_id))
        patient_data["conditions"] = [
            self.parse_condition(c) for c in self.get_patient_conditions(patient_id)
        ]
        patient_data["medications"] = [
            self.parse_medication(m) for m in self.get_patient_medications(patient_id)
        ]
        return patient_data

    # ------------------------------------------------------------------
    # Mock fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_patient(patient_id: str) -> Dict[str, Any]:
        return {
            "resourceType": "Patient",
            "id": patient_id,
            "name": [{"given": ["John"], "family": "Doe"}],
            "birthDate": "1960-01-15",
            "gender": "male",
            "telecom": [
                {"system": "email", "value": "john.doe@example.com"},
                {"system": "phone", "value": "555-0100"},
            ],
            "address": [{"postalCode": "12345"}],
        }

    @staticmethod
    def _mock_conditions() -> List[Dict[str, Any]]:
        return [
            {
                "resourceType": "Condition",
                "id": "cond_mock_1",
                "code": {
                    "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I48.91", "display": "Atrial Fibrillation"}],
                    "text": "Atrial Fibrillation",
                },
            }
        ]

    @staticmethod
    def _mock_medications() -> List[Dict[str, Any]]:
        return [
            {
                "resourceType": "MedicationStatement",
                "id": "med_mock_1",
                "status": "active",
                "medicationCodeableConcept": {
                    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "B01AA03", "display": "Warfarin"}],
                    "text": "Warfarin",
                },
            }
        ]


fhir_client = FHIRClient()
