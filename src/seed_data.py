"""
Seed the database with synthetic clinical trial data.

Generates:
  - 300 patients  (varied demographics, conditions, medications)
  - 25 trials     (Phase 1–4, diverse therapeutic areas)
  - ~2 500–3 500 patient-trial matches with rule + ML scores

Usage (standalone):
    DATABASE_URL=postgresql://... python -m src.seed_data

Usage (via API):
    POST /admin/seed
"""

import logging
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.models import Patient, Trial, PatientTrialMatch, SessionLocal
from src.eligibility import matcher
from src.ml_prediction import predictor, EnrollmentPredictor

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Reference data
# ------------------------------------------------------------------

_RNG = random.Random(42)

FIRST_NAMES_M = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth",
    "George", "Joshua", "Kevin", "Brian", "Edward", "Ronald", "Timothy",
    "Jason", "Jeffrey", "Ryan", "Gary", "Jacob", "Nicholas", "Eric",
    "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
]

FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Emily", "Dorothy", "Kimberly",
    "Carol", "Michelle", "Amanda", "Melissa", "Deborah", "Stephanie",
    "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen", "Amy",
    "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma",
    "Nicole", "Helen", "Samantha",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]

POSTAL_CODES = [
    "10001", "10002", "90210", "60601", "77001", "30301", "98101",
    "85001", "94101", "02101", "20001", "33101", "48201", "55401",
    "80201", "37201", "23219", "45202", "46204", "73102",
]

# condition key -> {icd10_code, display}
CONDITIONS = {
    "hypertension":              {"icd10_code": "I10",    "display": "Essential Hypertension"},
    "type2_diabetes":            {"icd10_code": "E11",    "display": "Type 2 Diabetes Mellitus"},
    "atrial_fibrillation":       {"icd10_code": "I48.91", "display": "Atrial Fibrillation"},
    "heart_failure":             {"icd10_code": "I50",    "display": "Heart Failure"},
    "ckd":                       {"icd10_code": "N18",    "display": "Chronic Kidney Disease"},
    "copd":                      {"icd10_code": "J44",    "display": "COPD"},
    "asthma":                    {"icd10_code": "J45",    "display": "Asthma"},
    "breast_cancer":             {"icd10_code": "C50",    "display": "Breast Cancer"},
    "prostate_cancer":           {"icd10_code": "C61",    "display": "Prostate Cancer"},
    "colorectal_cancer":         {"icd10_code": "C18",    "display": "Colorectal Cancer"},
    "lung_cancer":               {"icd10_code": "C34",    "display": "Lung Cancer"},
    "alzheimers":                {"icd10_code": "G30",    "display": "Alzheimer's Disease"},
    "depression":                {"icd10_code": "F32",    "display": "Major Depressive Disorder"},
    "obesity":                   {"icd10_code": "E66",    "display": "Obesity"},
    "stroke":                    {"icd10_code": "I63",    "display": "Ischemic Stroke"},
    "osteoporosis":              {"icd10_code": "M81",    "display": "Osteoporosis"},
    "nafld":                     {"icd10_code": "K76.0",  "display": "Non-Alcoholic Fatty Liver Disease"},
    "peripheral_artery_disease": {"icd10_code": "I73.9",  "display": "Peripheral Artery Disease"},
    "rheumatoid_arthritis":      {"icd10_code": "M05",    "display": "Rheumatoid Arthritis"},
    "multiple_sclerosis":        {"icd10_code": "G35",    "display": "Multiple Sclerosis"},
    "migraine":                  {"icd10_code": "G43",    "display": "Migraine"},
    "insomnia":                  {"icd10_code": "G47.00", "display": "Insomnia"},
    "pulmonary_hypertension":    {"icd10_code": "I27.0",  "display": "Pulmonary Arterial Hypertension"},
}

# medication key -> {code, display}
MEDICATIONS = {
    "metformin":         {"code": "A10BA02", "display": "Metformin"},
    "insulin":           {"code": "A10AB01", "display": "Insulin Glargine"},
    "lisinopril":        {"code": "C09AA01", "display": "Lisinopril"},
    "amlodipine":        {"code": "C08CA01", "display": "Amlodipine"},
    "atorvastatin":      {"code": "C10AA05", "display": "Atorvastatin"},
    "warfarin":          {"code": "B01AA03", "display": "Warfarin"},
    "apixaban":          {"code": "B01AF02", "display": "Apixaban"},
    "aspirin":           {"code": "N02BA01", "display": "Aspirin 81 mg"},
    "furosemide":        {"code": "C03CA01", "display": "Furosemide"},
    "metoprolol":        {"code": "C07AB02", "display": "Metoprolol Succinate"},
    "omeprazole":        {"code": "A02BC01", "display": "Omeprazole"},
    "sertraline":        {"code": "N06AB06", "display": "Sertraline"},
    "levothyroxine":     {"code": "H03AA01", "display": "Levothyroxine"},
    "albuterol":         {"code": "R03AC02", "display": "Albuterol Inhaler"},
    "prednisone":        {"code": "H02AB07", "display": "Prednisone"},
    "methotrexate":      {"code": "L01BA01", "display": "Methotrexate"},
    "adalimumab":        {"code": "L04AB04", "display": "Adalimumab"},
    "sumatriptan":       {"code": "N02CC01", "display": "Sumatriptan"},
    "topiramate":        {"code": "N03AX11", "display": "Topiramate"},
    "spironolactone":    {"code": "C03DA01", "display": "Spironolactone"},
    "rivaroxaban":       {"code": "B01AF01", "display": "Rivaroxaban"},
    "empagliflozin":     {"code": "A10BK03", "display": "Empagliflozin"},
    "semaglutide":       {"code": "A10BJ06", "display": "Semaglutide"},
    "bisoprolol":        {"code": "C07AB07", "display": "Bisoprolol"},
    "hydrochlorothiazide": {"code": "C03AA03", "display": "Hydrochlorothiazide"},
}

# condition -> typical medications (1–2 chosen at random)
CONDITION_MEDS = {
    "hypertension":              ["lisinopril", "amlodipine", "metoprolol", "hydrochlorothiazide", "bisoprolol"],
    "type2_diabetes":            ["metformin", "insulin", "empagliflozin", "semaglutide"],
    "atrial_fibrillation":       ["warfarin", "apixaban", "rivaroxaban", "metoprolol", "bisoprolol"],
    "heart_failure":             ["furosemide", "metoprolol", "lisinopril", "spironolactone", "bisoprolol"],
    "ckd":                       ["furosemide", "lisinopril", "spironolactone"],
    "copd":                      ["albuterol", "prednisone"],
    "asthma":                    ["albuterol", "prednisone"],
    "breast_cancer":             ["prednisone", "methotrexate"],
    "prostate_cancer":           ["prednisone"],
    "colorectal_cancer":         ["prednisone"],
    "lung_cancer":               ["prednisone"],
    "depression":                ["sertraline"],
    "rheumatoid_arthritis":      ["methotrexate", "adalimumab", "prednisone"],
    "migraine":                  ["sumatriptan", "topiramate"],
    "obesity":                   ["semaglutide"],
}

# condition key -> (prevalence_fraction, min_age, gender_restriction)
# prevalence is per-patient probability given age >= min_age
CONDITION_PROBS = [
    ("hypertension",              0.40, 35,  None),
    ("type2_diabetes",            0.18, 30,  None),
    ("atrial_fibrillation",       0.08, 50,  None),
    ("heart_failure",             0.05, 55,  None),
    ("ckd",                       0.12, 45,  None),
    ("copd",                      0.07, 45,  None),
    ("asthma",                    0.08, 18,  None),
    ("breast_cancer",             0.04, 40,  "female"),
    ("prostate_cancer",           0.06, 50,  "male"),
    ("colorectal_cancer",         0.02, 45,  None),
    ("lung_cancer",               0.02, 50,  None),
    ("alzheimers",                0.05, 65,  None),
    ("depression",                0.10, 18,  None),
    ("obesity",                   0.30, 18,  None),
    ("stroke",                    0.03, 55,  None),
    ("osteoporosis",              0.08, 55,  None),
    ("nafld",                     0.12, 30,  None),
    ("peripheral_artery_disease", 0.04, 50,  None),
    ("rheumatoid_arthritis",      0.03, 30,  None),
    ("multiple_sclerosis",        0.01, 20,  None),
    ("migraine",                  0.10, 18,  None),
    ("insomnia",                  0.12, 18,  None),
    ("pulmonary_hypertension",    0.01, 30,  None),
]

# ------------------------------------------------------------------
# Trial definitions
# ------------------------------------------------------------------

def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


TRIALS = [
    {
        "id": "TRIAL_AFIB_001",
        "name": "RHYTHM-GUARD: Novel Antiarrhythmic in Persistent AFib",
        "description": "Phase 3 RCT evaluating a new antiarrhythmic agent versus standard care in adults with persistent atrial fibrillation.",
        "sponsor": "CardioGen Therapeutics",
        "phase": "3",
        "primary_condition": "Atrial Fibrillation",
        "target_enrollment": 400,
        "start_date": _dt(2024, 1, 15),
        "completion_date": _dt(2026, 12, 31),
        "inclusion_criteria": [
            {"field": "condition:I48.91", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 80},
        ],
        "exclusion_criteria": [
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_DM2_001",
        "name": "GLYCO-CONTROL: Empagliflozin Optimisation in T2DM",
        "description": "Phase 3 study assessing add-on empagliflozin therapy in adults with poorly controlled Type 2 Diabetes Mellitus.",
        "sponsor": "MetaRx Inc.",
        "phase": "3",
        "primary_condition": "Type 2 Diabetes",
        "target_enrollment": 600,
        "start_date": _dt(2023, 6, 1),
        "completion_date": _dt(2026, 5, 31),
        "inclusion_criteria": [
            {"field": "condition:E11", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 30},
            {"field": "age", "operator": "LTE", "value": 75},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_HTN_001",
        "name": "PRESSURETRAK: Triple Therapy vs Dual in Resistant Hypertension",
        "description": "Phase 3 trial comparing triple versus dual antihypertensive regimens in patients with resistant hypertension.",
        "sponsor": "VascuPharm",
        "phase": "3",
        "primary_condition": "Hypertension",
        "target_enrollment": 500,
        "start_date": _dt(2024, 3, 1),
        "completion_date": _dt(2027, 2, 28),
        "inclusion_criteria": [
            {"field": "condition:I10", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 35},
            {"field": "age", "operator": "LTE", "value": 80},
        ],
        "exclusion_criteria": [
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_HF_001",
        "name": "HEARTSPAN: Sacubitril/Valsartan in HFpEF",
        "description": "Phase 2 study of sacubitril/valsartan in heart failure with preserved ejection fraction.",
        "sponsor": "CardioGen Therapeutics",
        "phase": "2",
        "primary_condition": "Heart Failure",
        "target_enrollment": 300,
        "start_date": _dt(2024, 6, 1),
        "completion_date": _dt(2026, 11, 30),
        "inclusion_criteria": [
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 45},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_CKD_001",
        "name": "NEPHRO-SHIELD: Finerenone in CKD with T2DM",
        "description": "Phase 3 trial of finerenone to slow CKD progression in patients with comorbid Type 2 Diabetes.",
        "sponsor": "RenalCare Biotech",
        "phase": "3",
        "primary_condition": "Chronic Kidney Disease",
        "target_enrollment": 450,
        "start_date": _dt(2023, 9, 1),
        "completion_date": _dt(2026, 8, 31),
        "inclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
            {"field": "condition:E11", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 40},
        ],
        "exclusion_criteria": [],
    },
    {
        "id": "TRIAL_COPD_001",
        "name": "AIRWAY-PLUS: Triple Inhaler in COPD GOLD 3–4",
        "description": "Phase 2 RCT of triple bronchodilator therapy vs standard dual therapy in severe COPD.",
        "sponsor": "RespiGen",
        "phase": "2",
        "primary_condition": "COPD",
        "target_enrollment": 250,
        "start_date": _dt(2024, 1, 1),
        "completion_date": _dt(2026, 6, 30),
        "inclusion_criteria": [
            {"field": "condition:J44", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 40},
            {"field": "age", "operator": "LTE", "value": 80},
        ],
        "exclusion_criteria": [
            {"field": "condition:C34", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_BC_001",
        "name": "IMMUNO-BREAST: Pembrolizumab in TNBC",
        "description": "Phase 2 study of pembrolizumab plus chemotherapy in triple-negative breast cancer.",
        "sponsor": "OncoBio Solutions",
        "phase": "2",
        "primary_condition": "Breast Cancer",
        "target_enrollment": 120,
        "start_date": _dt(2024, 4, 1),
        "completion_date": _dt(2027, 3, 31),
        "inclusion_criteria": [
            {"field": "condition:C50", "operator": "EXISTS", "value": None},
            {"field": "gender", "operator": "EQ", "value": "female"},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 70},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_PC_001",
        "name": "ANDROGEN-BLOCK: Enzalutamide in mCRPC",
        "description": "Phase 2 study of enzalutamide in metastatic castration-resistant prostate cancer.",
        "sponsor": "OncoBio Solutions",
        "phase": "2",
        "primary_condition": "Prostate Cancer",
        "target_enrollment": 150,
        "start_date": _dt(2024, 2, 1),
        "completion_date": _dt(2026, 10, 31),
        "inclusion_criteria": [
            {"field": "condition:C61", "operator": "EXISTS", "value": None},
            {"field": "gender", "operator": "EQ", "value": "male"},
            {"field": "age", "operator": "GTE", "value": 50},
        ],
        "exclusion_criteria": [
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_LC_001",
        "name": "LUNGPATH-1: Osimertinib in EGFR+ NSCLC",
        "description": "Phase 1/2 study of osimertinib in EGFR-mutant non-small cell lung cancer.",
        "sponsor": "TumorTarget Pharma",
        "phase": "1/2",
        "primary_condition": "Lung Cancer",
        "target_enrollment": 80,
        "start_date": _dt(2024, 7, 1),
        "completion_date": _dt(2027, 6, 30),
        "inclusion_criteria": [
            {"field": "condition:C34", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_ALZ_001",
        "name": "MEMGUARD: Anti-Amyloid Antibody in Early Alzheimer's",
        "description": "Phase 3 RCT of lecanemab in adults with mild cognitive impairment due to Alzheimer's disease.",
        "sponsor": "NeuroPharma Corp",
        "phase": "3",
        "primary_condition": "Alzheimer's Disease",
        "target_enrollment": 350,
        "start_date": _dt(2024, 3, 1),
        "completion_date": _dt(2027, 2, 28),
        "inclusion_criteria": [
            {"field": "condition:G30", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 55},
            {"field": "age", "operator": "LTE", "value": 85},
        ],
        "exclusion_criteria": [
            {"field": "condition:I63", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_DEP_001",
        "name": "BRIGHTMIND: Esketamine in Treatment-Resistant Depression",
        "description": "Phase 3 study of intranasal esketamine in adults with treatment-resistant MDD.",
        "sponsor": "MindBridge Pharma",
        "phase": "3",
        "primary_condition": "Major Depressive Disorder",
        "target_enrollment": 280,
        "start_date": _dt(2023, 11, 1),
        "completion_date": _dt(2026, 10, 31),
        "inclusion_criteria": [
            {"field": "condition:F32", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 65},
        ],
        "exclusion_criteria": [
            {"field": "condition:G30", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_OB_001",
        "name": "SLIMPATH: Semaglutide 2.4 mg in Class III Obesity",
        "description": "Phase 3 trial of high-dose semaglutide in adults with BMI ≥ 40 kg/m².",
        "sponsor": "MetaRx Inc.",
        "phase": "3",
        "primary_condition": "Obesity",
        "target_enrollment": 500,
        "start_date": _dt(2024, 1, 1),
        "completion_date": _dt(2026, 12, 31),
        "inclusion_criteria": [
            {"field": "condition:E66", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 70},
        ],
        "exclusion_criteria": [
            {"field": "condition:C50", "operator": "EXISTS", "value": None},
            {"field": "condition:C34", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_ASTHMA_001",
        "name": "AIRFLEX: Dupilumab in Moderate-Severe Asthma",
        "description": "Phase 2 study of dupilumab (anti-IL-4/IL-13) in adults with uncontrolled moderate-to-severe asthma.",
        "sponsor": "RespiGen",
        "phase": "2",
        "primary_condition": "Asthma",
        "target_enrollment": 200,
        "start_date": _dt(2024, 5, 1),
        "completion_date": _dt(2026, 4, 30),
        "inclusion_criteria": [
            {"field": "condition:J45", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 65},
        ],
        "exclusion_criteria": [
            {"field": "condition:J44", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_STROKE_001",
        "name": "CLOT-STOP: Ticagrelor vs Aspirin in Secondary Stroke Prevention",
        "description": "Phase 3 RCT comparing ticagrelor versus aspirin for secondary prevention after ischaemic stroke.",
        "sponsor": "VascuPharm",
        "phase": "3",
        "primary_condition": "Ischaemic Stroke",
        "target_enrollment": 350,
        "start_date": _dt(2024, 2, 1),
        "completion_date": _dt(2027, 1, 31),
        "inclusion_criteria": [
            {"field": "condition:I63", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 40},
            {"field": "age", "operator": "LTE", "value": 80},
        ],
        "exclusion_criteria": [
            {"field": "condition:I48.91", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_OST_001",
        "name": "BONEFORT: Romosozumab in Post-Menopausal Osteoporosis",
        "description": "Phase 3 study of romosozumab followed by denosumab to reduce fracture risk.",
        "sponsor": "SkeleGen",
        "phase": "3",
        "primary_condition": "Osteoporosis",
        "target_enrollment": 300,
        "start_date": _dt(2023, 10, 1),
        "completion_date": _dt(2026, 9, 30),
        "inclusion_criteria": [
            {"field": "condition:M81", "operator": "EXISTS", "value": None},
            {"field": "gender", "operator": "EQ", "value": "female"},
            {"field": "age", "operator": "GTE", "value": 55},
        ],
        "exclusion_criteria": [
            {"field": "condition:I10", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_NAFLD_001",
        "name": "LIVERWISE: Lanifibranor in NASH with Fibrosis",
        "description": "Phase 2b study of lanifibranor (pan-PPAR agonist) in NASH with stage 2–3 fibrosis.",
        "sponsor": "HepaGen Pharma",
        "phase": "2",
        "primary_condition": "Non-Alcoholic Fatty Liver Disease",
        "target_enrollment": 180,
        "start_date": _dt(2024, 4, 1),
        "completion_date": _dt(2026, 9, 30),
        "inclusion_criteria": [
            {"field": "condition:K76.0", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 25},
            {"field": "age", "operator": "LTE", "value": 70},
        ],
        "exclusion_criteria": [
            {"field": "condition:C18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_PAD_001",
        "name": "LIMBFLOW: Vonapanitase in Peripheral Artery Disease",
        "description": "Phase 2 study of vonapanitase to improve AV access in patients with PAD.",
        "sponsor": "VascuPharm",
        "phase": "2",
        "primary_condition": "Peripheral Artery Disease",
        "target_enrollment": 140,
        "start_date": _dt(2024, 6, 1),
        "completion_date": _dt(2026, 12, 31),
        "inclusion_criteria": [
            {"field": "condition:I73.9", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 45},
        ],
        "exclusion_criteria": [
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_RA_001",
        "name": "ARTHREX: Upadacitinib in Active RA",
        "description": "Phase 3 trial of upadacitinib (JAK1 inhibitor) in rheumatoid arthritis with inadequate DMARD response.",
        "sponsor": "ImmunoPath",
        "phase": "3",
        "primary_condition": "Rheumatoid Arthritis",
        "target_enrollment": 320,
        "start_date": _dt(2023, 8, 1),
        "completion_date": _dt(2026, 7, 31),
        "inclusion_criteria": [
            {"field": "condition:M05", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 75},
        ],
        "exclusion_criteria": [
            {"field": "condition:C50", "operator": "EXISTS", "value": None},
            {"field": "condition:C34", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_MS_001",
        "name": "NEUROSHIELD: Ofatumumab in Relapsing MS",
        "description": "Phase 3 study of subcutaneous ofatumumab in adults with relapsing-remitting multiple sclerosis.",
        "sponsor": "NeuroPharma Corp",
        "phase": "3",
        "primary_condition": "Multiple Sclerosis",
        "target_enrollment": 250,
        "start_date": _dt(2024, 1, 1),
        "completion_date": _dt(2027, 12, 31),
        "inclusion_criteria": [
            {"field": "condition:G35", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 55},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_MIG_001",
        "name": "HEADSAFE: Atogepant for Chronic Migraine Prevention",
        "description": "Phase 3 RCT of atogepant (oral CGRP antagonist) for chronic migraine prevention.",
        "sponsor": "MindBridge Pharma",
        "phase": "3",
        "primary_condition": "Migraine",
        "target_enrollment": 350,
        "start_date": _dt(2024, 3, 1),
        "completion_date": _dt(2026, 8, 31),
        "inclusion_criteria": [
            {"field": "condition:G43", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 65},
        ],
        "exclusion_criteria": [
            {"field": "condition:I63", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_INS_001",
        "name": "SLEEPWELL: Daridorexant in Chronic Insomnia",
        "description": "Phase 3 study of daridorexant (dual orexin receptor antagonist) in adults with chronic insomnia disorder.",
        "sponsor": "SleepPath Inc.",
        "phase": "3",
        "primary_condition": "Insomnia",
        "target_enrollment": 400,
        "start_date": _dt(2024, 2, 1),
        "completion_date": _dt(2026, 7, 31),
        "inclusion_criteria": [
            {"field": "condition:G47.00", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 22},
            {"field": "age", "operator": "LTE", "value": 70},
        ],
        "exclusion_criteria": [
            {"field": "condition:G30", "operator": "EXISTS", "value": None},
            {"field": "condition:F32", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_PAH_001",
        "name": "PULMO-EASE: Sotatercept in Pulmonary Arterial Hypertension",
        "description": "Phase 1/2 study of sotatercept in PAH patients on background therapy.",
        "sponsor": "PulmoThera",
        "phase": "1/2",
        "primary_condition": "Pulmonary Arterial Hypertension",
        "target_enrollment": 90,
        "start_date": _dt(2024, 5, 1),
        "completion_date": _dt(2027, 4, 30),
        "inclusion_criteria": [
            {"field": "condition:I27.0", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
            {"field": "age", "operator": "LTE", "value": 70},
        ],
        "exclusion_criteria": [
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_CRC_001",
        "name": "COLOBLOCK: Regorafenib in Metastatic Colorectal Cancer",
        "description": "Phase 2 study of regorafenib plus pembrolizumab in microsatellite-stable mCRC.",
        "sponsor": "TumorTarget Pharma",
        "phase": "2",
        "primary_condition": "Colorectal Cancer",
        "target_enrollment": 100,
        "start_date": _dt(2024, 8, 1),
        "completion_date": _dt(2027, 7, 31),
        "inclusion_criteria": [
            {"field": "condition:C18", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 18},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_DM2_HTN",
        "name": "CARDIO-DUAL: GLP-1 + SGLT-2 in T2DM with Hypertension",
        "description": "Phase 3 combination study of semaglutide + empagliflozin in patients with both T2DM and hypertension.",
        "sponsor": "MetaRx Inc.",
        "phase": "3",
        "primary_condition": "T2DM + Hypertension",
        "target_enrollment": 700,
        "start_date": _dt(2024, 1, 1),
        "completion_date": _dt(2027, 6, 30),
        "inclusion_criteria": [
            {"field": "condition:E11", "operator": "EXISTS", "value": None},
            {"field": "condition:I10", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 40},
            {"field": "age", "operator": "LTE", "value": 75},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
            {"field": "condition:I50", "operator": "EXISTS", "value": None},
        ],
    },
    {
        "id": "TRIAL_AFIB_HF",
        "name": "RHYTHM-HEART: Catheter Ablation in AFib-HF",
        "description": "Phase 3 study of pulmonary vein isolation vs medical therapy in patients with comorbid AFib and heart failure.",
        "sponsor": "CardioGen Therapeutics",
        "phase": "3",
        "primary_condition": "Atrial Fibrillation with Heart Failure",
        "target_enrollment": 200,
        "start_date": _dt(2024, 9, 1),
        "completion_date": _dt(2027, 8, 31),
        "inclusion_criteria": [
            {"field": "condition:I48.91", "operator": "EXISTS", "value": None},
            {"field": "condition:I50",    "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GTE", "value": 40},
            {"field": "age", "operator": "LTE", "value": 75},
        ],
        "exclusion_criteria": [
            {"field": "condition:N18", "operator": "EXISTS", "value": None},
        ],
    },
]


# ------------------------------------------------------------------
# Patient generation
# ------------------------------------------------------------------

def _age_to_dob(age: int) -> datetime:
    """Convert an integer age to an approximate date of birth."""
    base = datetime(2026, 4, 15, tzinfo=timezone.utc)
    return base - timedelta(days=age * 365 + _RNG.randint(0, 364))


def _random_age() -> int:
    """Age distribution weighted toward middle-age/elderly (trials focus here)."""
    bracket = _RNG.choices(
        [1, 2, 3, 4, 5],
        weights=[0.08, 0.15, 0.27, 0.32, 0.18],
    )[0]
    return {
        1: _RNG.randint(18, 29),
        2: _RNG.randint(30, 44),
        3: _RNG.randint(45, 59),
        4: _RNG.randint(60, 74),
        5: _RNG.randint(75, 89),
    }[bracket]


def _build_conditions(age: int, gender: str) -> list:
    conditions = []
    for ckey, prob, min_age, gender_req in CONDITION_PROBS:
        if age < min_age:
            continue
        if gender_req and gender != gender_req:
            continue
        if _RNG.random() < prob:
            cdef = CONDITIONS[ckey]
            conditions.append({
                "code": cdef["icd10_code"],
                "icd10_code": cdef["icd10_code"],
                "display": cdef["display"],
            })
    return conditions


def _build_medications(conditions: list) -> list:
    cond_codes = {c["icd10_code"] for c in conditions}
    meds_chosen: dict = {}
    for ckey, cdef in CONDITIONS.items():
        if cdef["icd10_code"] in cond_codes and ckey in CONDITION_MEDS:
            for mkey in _RNG.sample(CONDITION_MEDS[ckey], k=min(2, len(CONDITION_MEDS[ckey]))):
                if mkey not in meds_chosen:
                    mdef = MEDICATIONS[mkey]
                    meds_chosen[mkey] = {
                        "code": mdef["code"],
                        "medication_code": mdef["code"],
                        "display": mdef["display"],
                    }
    # small chance of a random background med
    if _RNG.random() < 0.3:
        mkey = _RNG.choice(list(MEDICATIONS.keys()))
        if mkey not in meds_chosen:
            mdef = MEDICATIONS[mkey]
            meds_chosen[mkey] = {"code": mdef["code"], "medication_code": mdef["code"], "display": mdef["display"]}
    return list(meds_chosen.values())


def _generate_patients(n: int = 300) -> list:
    patients = []
    for i in range(1, n + 1):
        gender = _RNG.choice(["male", "female"])
        first_name = _RNG.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
        last_name = _RNG.choice(LAST_NAMES)
        age = _random_age()
        dob = _age_to_dob(age)
        # spread created_at over the past 2 years for time-series charts
        created_offset = timedelta(days=_RNG.randint(0, 730))
        created_at = datetime(2024, 4, 1, tzinfo=timezone.utc) + created_offset

        conds = _build_conditions(age, gender)
        meds = _build_medications(conds)

        patients.append({
            "id": f"PT{i:04d}",
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": dob,
            "gender": gender,
            "email": f"{first_name.lower()}.{last_name.lower()}{i}@example.com",
            "phone": f"555-{_RNG.randint(1000, 9999)}",
            "postal_code": _RNG.choice(POSTAL_CODES),
            "conditions": conds,
            "medications": meds,
            "created_at": created_at,
        })
    return patients


# ------------------------------------------------------------------
# Core seeding logic
# ------------------------------------------------------------------

def run_seed(db: Session) -> dict:
    """Insert patients, trials, and matches. Returns a stats dict."""
    stats = {
        "patients_created": 0,
        "patients_skipped": 0,
        "trials_created": 0,
        "trials_skipped": 0,
        "matches_created": 0,
        "matches_skipped": 0,
        "eligible_matches": 0,
    }

    # ---- Patients ----
    logger.info("Seeding patients…")
    patient_rows = _generate_patients(300)
    for pdata in patient_rows:
        if db.query(Patient).filter(Patient.id == pdata["id"]).first():
            stats["patients_skipped"] += 1
            continue
        p = Patient(
            id=pdata["id"],
            first_name=pdata["first_name"],
            last_name=pdata["last_name"],
            date_of_birth=pdata["date_of_birth"],
            gender=pdata["gender"],
            email=pdata["email"],
            phone=pdata["phone"],
            postal_code=pdata["postal_code"],
            conditions=pdata["conditions"],
            medications=pdata["medications"],
            created_at=pdata["created_at"],
            updated_at=pdata["created_at"],
        )
        db.add(p)
        stats["patients_created"] += 1
    db.commit()
    logger.info(f"Patients: {stats['patients_created']} created, {stats['patients_skipped']} skipped.")

    # ---- Trials ----
    logger.info("Seeding trials…")
    for tdata in TRIALS:
        if db.query(Trial).filter(Trial.id == tdata["id"]).first():
            stats["trials_skipped"] += 1
            continue
        t = Trial(
            id=tdata["id"],
            name=tdata["name"],
            description=tdata["description"],
            sponsor=tdata["sponsor"],
            phase=tdata["phase"],
            primary_condition=tdata["primary_condition"],
            target_enrollment=tdata["target_enrollment"],
            inclusion_criteria=tdata["inclusion_criteria"],
            exclusion_criteria=tdata["exclusion_criteria"],
            start_date=tdata["start_date"],
            completion_date=tdata["completion_date"],
        )
        db.add(t)
        stats["trials_created"] += 1
    db.commit()
    logger.info(f"Trials: {stats['trials_created']} created, {stats['trials_skipped']} skipped.")

    # ---- Matches ----
    logger.info("Generating matches (300 patients × 25 trials)…")
    all_patients = db.query(Patient).all()
    all_trials = db.query(Trial).all()

    existing_pairs = {
        (m.patient_id, m.trial_id)
        for m in db.query(PatientTrialMatch.patient_id, PatientTrialMatch.trial_id).all()
    }

    batch: list = []
    BATCH_SIZE = 200

    for trial in all_trials:
        trial_dict = {
            "id": trial.id,
            "inclusion_criteria": trial.inclusion_criteria or [],
            "exclusion_criteria": trial.exclusion_criteria or [],
        }
        for patient in all_patients:
            if (patient.id, trial.id) in existing_pairs:
                stats["matches_skipped"] += 1
                continue

            patient_dict = {
                "id": patient.id,
                "first_name": patient.first_name,
                "last_name": patient.last_name,
                "date_of_birth": patient.date_of_birth,
                "gender": patient.gender,
                "conditions": patient.conditions or [],
                "medications": patient.medications or [],
            }

            # Rule-based eligibility
            rule = matcher.check_match(patient_dict, trial_dict)

            # ML enrollment prediction
            features = EnrollmentPredictor._dict_to_features(patient_dict)
            ml = predictor.predict(features, patient.id, trial.id)

            rule_score = float(rule["match_score"])
            ml_score = ml.enrollment_probability * 100.0
            combined = round(0.5 * rule_score + 0.5 * ml_score, 2)

            # Spread match created_at over 18 months for time-series richness
            match_created = patient.created_at + timedelta(
                days=_RNG.randint(0, min(540, (datetime.now(timezone.utc) - patient.created_at).days or 1))
            )

            # Simulate ~30% enrollment rate for eligible patients
            is_eligible = rule["eligible"]
            enrolled = is_eligible and (_RNG.random() < 0.30)

            match = PatientTrialMatch(
                patient_id=patient.id,
                trial_id=trial.id,
                rule_match_score=rule_score,
                ml_match_score=round(ml_score, 2),
                enrollment_probability=round(ml.enrollment_probability, 4),
                combined_score=combined,
                match_status="ELIGIBLE" if is_eligible else "INELIGIBLE",
                matched_criteria=rule["matched_inclusion"],
                violated_criteria=rule["violated_exclusion"],
                reasons=rule["reasons"],
                letter_sent=is_eligible,
                letter_sent_date=match_created + timedelta(days=_RNG.randint(1, 14)) if is_eligible else None,
                enrolled=enrolled,
                enrollment_date=match_created + timedelta(days=_RNG.randint(14, 60)) if enrolled else None,
                created_at=match_created,
                updated_at=match_created,
            )
            batch.append(match)
            stats["matches_created"] += 1
            if is_eligible:
                stats["eligible_matches"] += 1

            if len(batch) >= BATCH_SIZE:
                db.bulk_save_objects(batch)
                db.commit()
                batch.clear()
                logger.info(f"  …{stats['matches_created']} matches written so far")

    if batch:
        db.bulk_save_objects(batch)
        db.commit()

    logger.info(
        f"Matches: {stats['matches_created']} created "
        f"({stats['eligible_matches']} eligible, {stats['matches_skipped']} skipped)."
    )
    return stats


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with SessionLocal() as db:
        stats = run_seed(db)

    print("\nSeed complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
