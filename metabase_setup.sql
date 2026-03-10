CREATE MATERIALIZED VIEW enrollment_stats AS
SELECT 
  COUNT(*) as total_patients,
  COUNT(CASE WHEN trial_id IS NOT NULL THEN 1 END) as enrolled,
  ROUND(100.0 * COUNT(CASE WHEN trial_id IS NOT NULL THEN 1 END) / COUNT(*), 2) as enrollment_rate,
  AVG(age) as avg_age,
  STDDEV(age) as stddev_age,
  COUNT(CASE WHEN gender = 'M' THEN 1 END) as male_count,
  COUNT(CASE WHEN gender = 'F' THEN 1 END) as female_count
FROM patients;

CREATE MATERIALIZED VIEW trial_enrollment_breakdown AS
SELECT 
  t.id,
  t.name,
  COUNT(DISTINCT e.patient_id) as enrolled_count,
  t.target_enrollment,
  ROUND(100.0 * COUNT(DISTINCT e.patient_id) / t.target_enrollment, 2) as pct_complete
FROM trials t
LEFT JOIN enrollments e ON t.id = e.trial_id
GROUP BY t.id, t.name, t.target_enrollment
ORDER BY pct_complete DESC;

CREATE MATERIALIZED VIEW eligibility_failure_analysis AS
SELECT 
  em.exclusion_id,
  ex.description,
  COUNT(DISTINCT em.patient_id) as affected_patients,
  ROUND(100.0 * COUNT(DISTINCT em.patient_id) / (SELECT COUNT(*) FROM patients), 2) as pct_of_population
FROM eligibility_mismatches em
JOIN exclusion_criteria ex ON em.exclusion_id = ex.id
GROUP BY em.exclusion_id, ex.description
ORDER BY affected_patients DESC;

CREATE MATERIALIZED VIEW ml_prediction_stats AS
SELECT 
  COUNT(*) as total_predictions,
  ROUND(AVG(enrollment_probability), 4) as avg_prob,
  ROUND(MIN(enrollment_probability), 4) as min_prob,
  ROUND(MAX(enrollment_probability), 4) as max_prob,
  COUNT(CASE WHEN enrollment_probability > 0.7 THEN 1 END) as high_prob_count,
  COUNT(CASE WHEN enrollment_probability BETWEEN 0.4 AND 0.7 THEN 1 END) as medium_prob_count,
  COUNT(CASE WHEN enrollment_probability < 0.4 THEN 1 END) as low_prob_count
FROM ml_predictions;

CREATE MATERIALIZED VIEW patient_cohort_features AS
SELECT 
  COUNT(*) as cohort_size,
  ROUND(AVG(age), 1) as avg_age,
  ROUND(AVG(CAST(gender = 'M' AS INT)), 3) as male_ratio,
  ROUND(AVG(num_conditions), 2) as avg_conditions,
  ROUND(AVG(num_medications), 2) as avg_medications,
  ROUND(AVG(bmi), 2) as avg_bmi,
  COUNT(CASE WHEN smoker THEN 1 END) as smoker_count,
  COUNT(CASE WHEN prior_trial_participation THEN 1 END) as prior_trial_count
FROM patients;

CREATE INDEX idx_enrollments_trial ON enrollments(trial_id);
CREATE INDEX idx_enrollments_patient ON enrollments(patient_id);
CREATE INDEX idx_ml_predictions_patient ON ml_predictions(patient_id);
CREATE INDEX idx_eligibility_mismatches_patient ON eligibility_mismatches(patient_id);
