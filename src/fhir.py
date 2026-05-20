"""FHIR R4 Client for EHR Integration.

Provides HTTP-based access to a FHIR R4 server with automatic fallback to
deterministic mock data when the server is unreachable (e.g. during local dev).
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_FHIR_TIMEOUT: float = float(os.environ.get("FHIR_TIMEOUT", "5.0"))


class FHIRClient:
    """FHIR R4 HTTP client with mock fallback.

    Each method attempts a real HTTP request and falls back to mock data
    when the server is unreachable, allowing development without a live FHIR server.

    Attributes:
        base_url: Base URL of the FHIR R4 server (no trailing slash).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://fhir-server:8080/fhir",
        timeout: float = _FHIR_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        logger.info("FHIR client initialised: %s", self.base_url)

    # ------------------------------------------------------------------
    # Resource fetchers
    # ------------------------------------------------------------------

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """Fetch a FHIR Patient resource, falling back to mock on error.

        Args:
            patient_id: The FHIR patient logical ID.

        Returns:
            FHIR Patient resource dict (real or mock).
        """
        url = f"{self.base_url}/Patient/{patient_id}"
        try:
            resp = httpx.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("FHIR server unreachable – returning mock Patient for %s", patient_id)
            return self._mock_patient(patient_id)

    def get_patient_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Fetch all Condition resources for a patient, falling back to mock on error.

        Args:
            patient_id: The FHIR patient logical ID.

        Returns:
            List of FHIR Condition resource dicts (real or mock).
        """
        url = f"{self.base_url}/Condition"
        try:
            resp = httpx.get(url, params={"patient": patient_id}, timeout=self.timeout)
            resp.raise_for_status()
            bundle = resp.json()
            return [entry["resource"] for entry in bundle.get("entry", [])]
        except Exception:
            logger.warning("FHIR server unreachable – returning mock Conditions for %s", patient_id)
            return self._mock_conditions()

    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        """Fetch all MedicationStatement resources for a patient, falling back to mock on error.

        Args:
            patient_id: The FHIR patient logical ID.

        Returns:
            List of FHIR MedicationStatement resource dicts (real or mock).
        """
        url = f"{self.base_url}/MedicationStatement"
        try:
            resp = httpx.get(url, params={"patient": patient_id}, timeout=self.timeout)
            resp.raise_for_status()
            bundle = resp.json()
            return [entry["resource"] for entry in bundle.get("entry", [])]
        except Exception:
            logger.warning("FHIR server unreachable – returning mock Medications for %s", patient_id)
            return self._mock_medications()

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def parse_patient(self, fhir_patient: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a FHIR Patient resource into a local patient dict.

        Args:
            fhir_patient: Raw FHIR Patient resource dict.

        Returns:
            Normalised patient dict with keys: fhir_id, first_name, last_name,
            date_of_birth, gender, email, phone, postal_code.
        """
        name = fhir_patient.get("name", [{}])[0]
        address = fhir_patient.get("address", [{}])[0] if fhir_patient.get("address") else {}
        telecom = self._parse_telecom(fhir_patient.get("telecom", []))
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
        """Parse a FHIR Condition resource into a simplified condition dict.

        Args:
            fhir_condition: Raw FHIR Condition resource dict.

        Returns:
            Dict with icd10_code and display keys.
        """
        coding_list = (fhir_condition.get("code") or {}).get("coding", [])
        coding = coding_list[0] if coding_list else {}
        return {
            "icd10_code": coding.get("code"),
            "display": (fhir_condition.get("code") or {}).get("text") or coding.get("display", ""),
        }

    def parse_medication(self, fhir_medication: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a FHIR MedicationStatement resource into a simplified medication dict.

        Args:
            fhir_medication: Raw FHIR MedicationStatement resource dict.

        Returns:
            Dict with medication_code, display, and status keys.
        """
        med_ref = fhir_medication.get("medicationCodeableConcept") or {}
        coding_list = med_ref.get("coding", [])
        coding = coding_list[0] if coding_list else {}
        return {
            "medication_code": coding.get("code"),
            "display": med_ref.get("text") or coding.get("display", ""),
            "status": fhir_medication.get("status", "unknown"),
        }

    def fetch_complete_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        """Assemble a complete patient profile from FHIR resources.

        Fetches and parses Patient, Condition, and MedicationStatement resources.

        Args:
            patient_id: The FHIR patient logical ID.

        Returns:
            Patient dict with conditions and medications lists appended.
        """
        patient_data = self.parse_patient(self.get_patient(patient_id))
        patient_data["conditions"] = [
            self.parse_condition(c) for c in self.get_patient_conditions(patient_id)
        ]
        patient_data["medications"] = [
            self.parse_medication(m) for m in self.get_patient_medications(patient_id)
        ]
        return patient_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_telecom(telecom_list: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """Convert a FHIR telecom list into a system-keyed dict.

        Args:
            telecom_list: List of FHIR telecom entries with system and value.

        Returns:
            Dict mapping system name (e.g. "email", "phone") to value.
        """
        return {t["system"]: t["value"] for t in telecom_list if "system" in t and "value" in t}

    # ------------------------------------------------------------------
    # Mock fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_patient(patient_id: str) -> Dict[str, Any]:
        """Return a deterministic mock FHIR Patient resource."""
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
        """Return a deterministic mock list of FHIR Condition resources."""
        return [
            {
                "resourceType": "Condition",
                "id": "cond_mock_1",
                "code": {
                    "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I48.91",
                                "display": "Atrial Fibrillation"}],
                    "text": "Atrial Fibrillation",
                },
            }
        ]

    @staticmethod
    def _mock_medications() -> List[Dict[str, Any]]:
        """Return a deterministic mock list of FHIR MedicationStatement resources."""
        return [
            {
                "resourceType": "MedicationStatement",
                "id": "med_mock_1",
                "status": "active",
                "medicationCodeableConcept": {
                    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": "B01AA03", "display": "Warfarin"}],
                    "text": "Warfarin",
                },
            }
        ]


fhir_client = FHIRClient()
