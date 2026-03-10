import numpy as np
import pickle
import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class PatientFeatures:
    age: float = 0.0
    gender_male: int = 0
    num_conditions: int = 0
    num_medications: int = 0
    has_diabetes: int = 0
    has_hypertension: int = 0
    has_heart_disease: int = 0
    has_cancer: int = 0
    has_afib: int = 0
    smoker: int = 0
    bmi: float = 25.0
    prior_trial_participation: int = 0
    distance_to_site_km: float = 50.0
    num_exclusion_flags: int = 0

    def to_array(self):
        return np.array([self.age,self.gender_male,self.num_conditions,self.num_medications,self.has_diabetes,self.has_hypertension,self.has_heart_disease,self.has_cancer,self.has_afib,self.smoker,self.bmi,self.prior_trial_participation,self.distance_to_site_km,self.num_exclusion_flags],dtype=np.float32)

    @staticmethod
    def feature_names():
        return ["age","gender_male","num_conditions","num_medications","has_diabetes","has_hypertension","has_heart_disease","has_cancer","has_afib","smoker","bmi","prior_trial_participation","distance_to_site_km","num_exclusion_flags"]

@dataclass
class PredictionResult:
    patient_id: str
    trial_id: str
    enrollment_probability: float
    predicted_enrolled: bool
    confidence: str
    key_factors: List[Dict[str,Any]] = field(default_factory=list)
    recommendation: str = ""
    predicted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

def _gen_data(n=500):
    rng = np.random.default_rng(42)
    ages=rng.uniform(18,80,n); gm=rng.integers(0,2,n); nc=rng.integers(0,8,n)
    nm=rng.integers(0,10,n); diab=rng.integers(0,2,n); htn=rng.integers(0,2,n)
    hd=rng.integers(0,2,n); ca=rng.integers(0,2,n); af=rng.integers(0,2,n)
    sm=rng.integers(0,2,n); bmi=rng.uniform(17,45,n); pt=rng.integers(0,2,n)
    dist=rng.uniform(0,200,n); ef=rng.integers(0,5,n)
    score=(0.3*((ages>=30)&(ages<=70)).astype(float)+0.1*(nc<5).astype(float)+0.1*(nm<7).astype(float)-0.2*ca-0.15*(ef>1).astype(float)-0.1*(dist>100).astype(float)+0.1*pt+rng.normal(0,0.1,n))
    y=(score>0.2).astype(int)
    X=np.column_stack([ages,gm,nc,nm,diab,htn,hd,ca,af,sm,bmi,pt,dist,ef]).astype(np.float32)
    return X,y

MODEL_PATH=os.path.join(os.path.dirname(__file__),"enrollment_model.pkl")

class EnrollmentPredictor:
    def __init__(self):
        self.model=None
        self.feature_names=PatientFeatures.feature_names()
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH,"rb") as f: self.model=pickle.load(f)
        else: self._train()

    def _train(self):
        try: import xgboost as xgb
        except ImportError: return
        X,y=_gen_data(1000); s=int(0.8*len(X))
        self.model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=42)
        self.model.fit(X[:s],y[:s],eval_set=[(X[s:],y[s:])],verbose=False)
        with open(MODEL_PATH,"wb") as f: pickle.dump(self.model,f)

    def predict(self,features,patient_id,trial_id):
        x=features.to_array().reshape(1,-1)
        prob=float(self.model.predict_proba(x)[0][1]) if self.model else self._rule(x[0])
        conf="HIGH" if (prob>=0.75 or prob<=0.25) else "MEDIUM" if (prob>=0.60 or prob<=0.40) else "LOW"
        return PredictionResult(patient_id=patient_id,trial_id=trial_id,enrollment_probability=round(prob,4),predicted_enrolled=prob>=0.5,confidence=conf,key_factors=self._explain(features),recommendation=self._rec(prob,features))

    def _rule(self,x):
        d=dict(zip(self.feature_names,x)); s=0.5
        if 30<=d["age"]<=70: s+=0.10
        if d["num_conditions"]>5: s-=0.10
        if d["has_cancer"]==1: s-=0.20
        if d["num_exclusion_flags"]>1: s-=0.15*d["num_exclusion_flags"]
        if d["distance_to_site_km"]>100: s-=0.10
        if d["prior_trial_participation"]==1: s+=0.10
        return float(np.clip(s,0,1))

    def _explain(self,f):
        factors=[]
        if 30<=f.age<=70: factors.append({"factor":"Age in optimal range","impact":"positive"})
        elif f.age>75: factors.append({"factor":"Age above 75","impact":"negative"})
        if f.has_cancer: factors.append({"factor":"Cancer diagnosis","impact":"negative"})
        if f.has_afib: factors.append({"factor":"AFib present","impact":"positive"})
        if f.num_exclusion_flags>0: factors.append({"factor":f"{f.num_exclusion_flags} exclusion flag(s)","impact":"negative"})
        if f.prior_trial_participation: factors.append({"factor":"Prior trial participation","impact":"positive"})
        if f.distance_to_site_km>100: factors.append({"factor":f"Distance {f.distance_to_site_km:.0f}km","impact":"negative"})
        if self.model and hasattr(self.model,"feature_importances_"):
            imp=self.model.feature_importances_
            for i in np.argsort(imp)[::-1][:3]:
                factors.append({"factor":f"ML:{self.feature_names[i]}","impact":"model","importance":round(float(imp[i]),4)})
        return factors[:6]

    def _rec(self,prob,f):
        if prob>=0.75: return "Strong candidate – prioritise outreach."
        if prob>=0.55: return "Likely eligible – schedule screening."
        if prob>=0.40: return "Borderline – review with PI."
        if f.num_exclusion_flags>2: return "Multiple exclusion flags – confirm with PI."
        return "Low probability – consider future trials."

    def predict_batch(self,patients,trial_id):
        results=[self.predict(self._dict_to_features(p),p.get("id","unknown"),trial_id) for p in patients]
        return sorted(results,key=lambda r:r.enrollment_probability,reverse=True)

    @staticmethod
    def _dict_to_features(p):
        dob=p.get("date_of_birth")
        if isinstance(dob,(date,datetime)): age=(datetime.now().date()-(dob.date() if isinstance(dob,datetime) else dob)).days/365.25
        elif isinstance(dob,str):
            try: age=(datetime.now()-datetime.fromisoformat(dob.replace("Z","+00:00")).replace(tzinfo=None)).days/365.25
            except: age=float(p.get("age",50))
        else: age=float(p.get("age",50))
        conds=p.get("conditions",[]) or []; meds=p.get("medications",[]) or []
        def has(lst,*kw): j=" ".join(str(i).lower() for i in lst); return int(any(k in j for k in kw))
        return PatientFeatures(age=age,gender_male=1 if str(p.get("gender","")).upper()=="MALE" else 0,num_conditions=len(conds),num_medications=len(meds),has_diabetes=has(conds,"diabetes"),has_hypertension=has(conds,"hypertension"),has_heart_disease=has(conds,"heart disease","coronary"),has_cancer=has(conds,"cancer","carcinoma"),has_afib=has(conds,"atrial fibrillation","afib"),smoker=int(p.get("smoker",False)),bmi=float(p.get("bmi",25)),prior_trial_participation=int(p.get("prior_trial_participation",False)),distance_to_site_km=float(p.get("distance_to_site_km",50)),num_exclusion_flags=int(p.get("num_exclusion_flags",0)))

def create_ml_router():
    from fastapi import APIRouter
    from pydantic import BaseModel
    router=APIRouter(prefix="/ml",tags=["ML"])
    predictor=EnrollmentPredictor()

    class SingleReq(BaseModel):
        patient_id: str; trial_id: str; features: Dict[str,Any]

    class BatchReq(BaseModel):
        patients: List[Dict[str,Any]]; trial_id: str

    @router.post("/predict")
    def predict(req: SingleReq):
        return predictor.predict(EnrollmentPredictor._dict_to_features(req.features),req.patient_id,req.trial_id).__dict__

    @router.post("/predict/batch")
    def batch(req: BatchReq):
        return [r.__dict__ for r in predictor.predict_batch(req.patients,req.trial_id)]

    @router.get("/model/info")
    def info():
        d={"features":PatientFeatures.feature_names(),"model_type":"XGBoost" if predictor.model else "RuleBased"}
        if predictor.model and hasattr(predictor.model,"feature_importances_"):
            d["feature_importances"]={n:round(float(i),4) for n,i in zip(PatientFeatures.feature_names(),predictor.model.feature_importances_)}
        return d

    return router
