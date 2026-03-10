# Clinical Trial Cohort Matching System

**AI-powered patient-trial matching platform** using FastAPI, PostgreSQL, XGBoost ML, and Metabase analytics.

## 🎯 Project Overview

Automates clinical trial patient recruitment through:
- Eligibility rule matching (inclusion/exclusion criteria)
- NLP clinical profile extraction from notes
- XGBoost enrollment probability prediction
- FHIR-compliant data representation
- Real-time analytics dashboard
- Automated patient outreach engine

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server (8000)                │
├──────────────┬──────────────┬──────────────────────────┤
│   CRUD Ops   │  Eligibility │  ML Predictions          │
│  Patients    │   Matching   │  XGBoost Model           │
│  Trials      │  NLP Extract │  Batch Scoring           │
├──────────────┴──────────────┴──────────────────────────┤
│          PostgreSQL Database (5432)                      │
├──────────────┬──────────────┬──────────────────────────┤
│   Patients   │   Trials     │  ML Predictions          │
│  Enrollments │  Criteria    │  Eligibility Matches     │
└──────────────┴──────────────┴──────────────────────────┘
       ↓
┌──────────────────────────┐
│  Metabase Dashboard      │
│  (3000) Analytics        │
└──────────────────────────┘
```

## 📋 Features

### Step 1-2: Core Backend
- FastAPI REST API with async/await
- SQLAlchemy ORM + PostgreSQL
- CRUD endpoints for patients, trials, enrollments

### Step 3: Eligibility Matching
- Rule-based matching engine
- Inclusion/exclusion criteria evaluation
- Cohort analysis

### Step 4: NLP Processing
- spaCy clinical entity extraction
- Medical abbreviation handling
- Clinical note parsing

### Step 5: FHIR Integration
- Mock FHIR R4 data generation
- HL7 FHIR Bundle serialization
- Patient/Observation resources

### Step 6: ML Predictions
- XGBoost classifier (100 estimators)
- 14 patient features
- Enrollment probability scoring
- Feature importance analysis

### Step 7: Metabase Dashboard
- Real-time analytics views
- Enrollment progress tracking
- Eligibility barrier analysis
- ML prediction distribution

### Step 8: Automated Recruitment
- ML-scored patient identification
- SMTP email integration
- Batch outreach automation
- Match score thresholds

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 4GB RAM minimum

### Setup

```bash
# Clone repo
git clone https://github.com/yourusername/clinical-trial-cohort
cd clinical-trial-cohort

# Start services
docker-compose up -d

# Wait for initialization (30s)
sleep 30

# Verify API
curl http://localhost:8000/health

# Verify DB
docker exec clinicaltrialcohort-trial_postgres-1 psql -U trialmatch -d trial_db -c "SELECT 1;"
```

### Seed Sample Data

```bash
# Create 5 sample patients
docker exec clinicaltrialcohort-trial_postgres-1 psql -U trialmatch -d trial_db << 'EOF'
INSERT INTO patients (age, gender, num_conditions, num_medications, has_diabetes, has_hypertension, has_heart_disease, has_cancer, has_afib, smoker, bmi, prior_trial_participation, distance_to_site_km)
VALUES
(52, 'M', 3, 4, true, true, false, false, false, true, 28.5, false, 12.3),
(48, 'F', 2, 3, false, true, false, false, false, false, 24.2, true, 8.5),
(65, 'M', 4, 5, true, true, true, false, true, false, 29.1, false, 20.0),
(41, 'F', 1, 2, false, false, false, false, false, true, 22.8, false, 5.5),
(58, 'M', 2, 3, true, false, false, false, false, false, 26.9, true, 15.2);
EOF
```

## 🔗 API Endpoints

### Patients
```bash
GET  /patients                    # List all patients
POST /patients                    # Create patient
GET  /patients/{patient_id}       # Get patient details
```

### Trials
```bash
GET  /trials                      # List all trials
POST /trials                      # Create trial
GET  /trials/{trial_id}          # Get trial details
```

### Eligibility Matching
```bash
POST /match/{patient_id}/{trial_id}   # Check eligibility
GET  /matches/{trial_id}              # Get all matches for trial
```

### ML Predictions
```bash
GET  /ml/model/info              # Model metadata
POST /ml/predict                 # Single prediction
POST /ml/predict/batch           # Batch predictions
```

### NLP Processing
```bash
POST /nlp/extract-entities       # Extract clinical entities
POST /nlp/clinical-profile       # Generate patient profile
```

## 📊 Analytics Dashboard

Access Metabase at **http://localhost:3000**

**Login:**
- Email: admin@trial.local
- Password: changeme123

**Dashboard Cards:**
- Enrollment rate (%)
- Trial progress tracking
- Top eligibility barriers
- ML prediction distribution
- Patient demographics

## 📁 Project Structure

```
clinical-trial-cohort/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── models.py            # SQLAlchemy ORM
│   ├── schemas.py           # Pydantic schemas
│   ├── database.py          # DB connection
│   ├── eligibility.py       # Matching logic
│   ├── nlp.py               # NLP extraction
│   ├── fhir.py              # FHIR generation
│   ├── ml_prediction.py     # XGBoost model
│   └── recruitment.py       # Recruitment engine
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.104.1 |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy 2.0 |
| ML | XGBoost 2.0, scikit-learn 1.3 |
| NLP | spaCy 3.7 |
| Analytics | Metabase 0.49 |
| Container | Docker, Docker Compose |

## 📈 ML Model

**XGBoost Classifier**
- Estimators: 100
- Max depth: 4
- Learning rate: 0.1

**Features (14):**
1. age
2. gender_male
3. num_conditions
4. num_medications
5. has_diabetes
6. has_hypertension
7. has_heart_disease
8. has_cancer
9. has_afib
10. smoker
11. bmi
12. prior_trial_participation
13. distance_to_site_km
14. num_exclusion_flags

**Feature Importance:**
- Cancer: 19.3%
- Age: 13.9%
- Prior participation: 10.8%
- Exclusion flags: 9.7%

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# API docs
curl http://localhost:8000/docs

# List patients
curl http://localhost:8000/patients

# Get ML model info
curl http://localhost:8000/ml/model/info

# Batch prediction
curl -X POST http://localhost:8000/ml/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"age": 52, "gender_male": 1, ...}]'
```

## 📊 Database Schema

### patients
```sql
CREATE TABLE patients (
  id SERIAL PRIMARY KEY,
  age INT,
  gender CHAR(1),
  num_conditions INT,
  num_medications INT,
  has_diabetes BOOLEAN,
  has_hypertension BOOLEAN,
  has_heart_disease BOOLEAN,
  has_cancer BOOLEAN,
  has_afib BOOLEAN,
  smoker BOOLEAN,
  bmi DECIMAL,
  prior_trial_participation BOOLEAN,
  distance_to_site_km DECIMAL,
  trial_id INT
);
```

### trials
```sql
CREATE TABLE trials (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255),
  description TEXT,
  target_enrollment INT
);
```

### enrollments
```sql
CREATE TABLE enrollments (
  patient_id INT REFERENCES patients(id),
  trial_id INT REFERENCES trials(id),
  PRIMARY KEY (patient_id, trial_id)
);
```

## 🔐 Security

- Database credentials in docker-compose (dev only)
- HTTPS ready (Uvicorn + nginx in production)
- SQL injection prevention (SQLAlchemy parameterized queries)
- CORS configured for frontend integration

## 📝 License

MIT License - See LICENSE file

## 👥 Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## 📞 Support

For issues, please open a GitHub issue or contact the maintainers.

---

**Built with ❤️ for clinical research automation**
