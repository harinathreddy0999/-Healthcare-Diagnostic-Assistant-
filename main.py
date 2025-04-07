from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from typing import List, Optional
import os
from dotenv import load_dotenv
import jwt
import bcrypt
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Healthcare Diagnostic Assistant API",
    description="API for managing patients, diagnoses, and appointments",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./healthcare.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    date_of_birth = Column(DateTime)
    address = Column(Text)
    medical_history = Column(Text)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

class AIService(Base):
    __tablename__ = "ai_services"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(Text)
    category = Column(String)
    status = Column(String)
    last_used = Column(DateTime)

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, index=True)
    doctor_id = Column(Integer, index=True)
    date_time = Column(DateTime)
    status = Column(String)  # Scheduled, Completed, Cancelled
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Additional Pydantic Models
class PatientCreate(BaseModel):
    name: str
    email: str
    phone: str
    date_of_birth: datetime
    address: str
    medical_history: str

class PatientResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: str
    date_of_birth: datetime
    address: str
    medical_history: str
    created_at: datetime

class AppointmentCreate(BaseModel):
    patient_id: int
    date_time: datetime
    notes: Optional[str] = None

class AppointmentResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    date_time: datetime
    status: str
    notes: Optional[str]
    created_at: datetime

# Dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Healthcare Diagnostic Assistant API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        name=user.name,
        password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
async def login(form_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.email).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "created_at": current_user.created_at
    }

# Patient Routes
@app.post("/patients/", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_patient = Patient(
        **patient.dict(),
        doctor_id=current_user.id
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/patients/", response_model=List[PatientResponse])
async def get_patients(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return patients

@app.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

# Appointment Routes
@app.post("/appointments/", response_model=AppointmentResponse)
async def create_appointment(
    appointment: AppointmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_appointment = Appointment(
        **appointment.dict(),
        doctor_id=current_user.id,
        status="Scheduled"
    )
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

@app.get("/appointments/", response_model=List[AppointmentResponse])
async def get_appointments(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    appointments = db.query(Appointment).filter(
        Appointment.doctor_id == current_user.id
    ).offset(skip).limit(limit).all()
    return appointments

@app.put("/appointments/{appointment_id}/status")
async def update_appointment_status(
    appointment_id: int,
    status: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appointment.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this appointment")
    
    appointment.status = status
    db.commit()
    return {"message": "Appointment status updated successfully"}

# AI Service Models
class ImageAnalysisRequest(BaseModel):
    image_type: str

class DiagnosisRequest(BaseModel):
    symptoms: List[str]
    patient_age: int
    patient_gender: str

class MonitoringRequest(BaseModel):
    patient_id: int
    vital_type: str
    value: float

# AI Service Routes
@app.post("/services/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    image_type: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert image to grayscale and normalize
        image = image.convert('L')
        image_array = np.array(image)
        image_array = image_array / 255.0
        
        # Analyze based on image type
        analysis_result = ""
        confidence = 0.95
        
        if image_type == "x-ray":
            # Simulate X-ray analysis
            if np.mean(image_array) < 0.5:
                analysis_result = "Analysis shows possible bone density issues. Recommend further examination."
                confidence = 0.85
            else:
                analysis_result = "Analysis shows normal bone structure. No fractures detected."
        
        elif image_type == "mri":
            # Simulate MRI analysis
            if np.std(image_array) > 0.2:
                analysis_result = "Detected unusual tissue patterns. Further examination recommended."
                confidence = 0.88
            else:
                analysis_result = "Brain tissue appears normal. No abnormalities detected."
        
        elif image_type == "ct":
            # Simulate CT scan analysis
            if np.max(image_array) > 0.9:
                analysis_result = "Possible areas of concern detected. Recommend follow-up."
                confidence = 0.82
            else:
                analysis_result = "Chest CT scan shows clear lung fields."
        
        else:
            analysis_result = "Analysis complete. No significant abnormalities detected."
        
        return {
            "status": "success",
            "analysis": analysis_result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "image_stats": {
                "mean_intensity": float(np.mean(image_array)),
                "std_intensity": float(np.std(image_array)),
                "max_intensity": float(np.max(image_array))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/services/diagnose")
async def diagnose_symptoms(
    request: DiagnosisRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Convert symptoms to feature vector
        symptoms_list = [
            "fever", "cough", "fatigue", "shortness_of_breath", 
            "headache", "body_ache", "sore_throat", "congestion"
        ]
        features = np.zeros(len(symptoms_list))
        for symptom in request.symptoms:
            if symptom.lower() in symptoms_list:
                features[symptoms_list.index(symptom.lower())] = 1
        
        # Add age and gender features
        features = np.append(features, [request.patient_age / 100.0])
        features = np.append(features, [1.0 if request.patient_gender.lower() == "male" else 0.0])
        
        # Simulate diagnosis model
        if "fever" in request.symptoms and "cough" in request.symptoms:
            diagnosis = "Possible respiratory infection"
            confidence = 0.85
            recommendations = [
                "Rest and hydration",
                "Over-the-counter fever reducer",
                "Monitor symptoms",
                "Consult doctor if symptoms worsen"
            ]
        elif "headache" in request.symptoms and "fatigue" in request.symptoms:
            diagnosis = "Possible stress-related condition or migraine"
            confidence = 0.75
            recommendations = [
                "Rest in a quiet, dark room",
                "Stay hydrated",
                "Consider stress management techniques",
                "Consult doctor if symptoms persist"
            ]
        else:
            diagnosis = "Mild condition, possibly viral"
            confidence = 0.70
            recommendations = [
                "Rest and hydration",
                "Monitor symptoms",
                "Over-the-counter symptom relief"
            ]
        
        return {
            "status": "success",
            "diagnosis": diagnosis,
            "confidence": confidence,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/services/monitor-vitals")
async def monitor_vitals(
    request: MonitoringRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Simulate vital signs analysis
        vital_ranges = {
            "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
            "blood_pressure_systolic": {"min": 90, "max": 120, "unit": "mmHg"},
            "blood_pressure_diastolic": {"min": 60, "max": 80, "unit": "mmHg"},
            "temperature": {"min": 36.5, "max": 37.5, "unit": "°C"},
            "oxygen_saturation": {"min": 95, "max": 100, "unit": "%"},
            "respiratory_rate": {"min": 12, "max": 20, "unit": "breaths/min"}
        }
        
        vital_type = request.vital_type.lower()
        value = request.value
        
        if vital_type in vital_ranges:
            range_info = vital_ranges[vital_type]
            is_normal = range_info["min"] <= value <= range_info["max"]
            reference_range = f"{range_info['min']}-{range_info['max']} {range_info['unit']}"
            
            if is_normal:
                recommendation = f"{vital_type.replace('_', ' ').title()} is within normal range"
            else:
                if value < range_info["min"]:
                    recommendation = f"Low {vital_type.replace('_', ' ')}. Medical attention may be required."
                else:
                    recommendation = f"High {vital_type.replace('_', ' ')}. Medical attention may be required."
        else:
            raise HTTPException(status_code=400, detail="Unsupported vital type")
        
        return {
            "status": "success",
            "vital_type": vital_type,
            "value": value,
            "unit": range_info["unit"],
            "is_normal": is_normal,
            "reference_range": reference_range,
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/services/generate-report")
async def generate_report(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get patient data
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Generate comprehensive report
        report = {
            "patient_info": {
                "name": patient.name,
                "date_of_birth": patient.date_of_birth.isoformat(),
                "medical_history": patient.medical_history
            },
            "vital_signs": {
                "blood_pressure": "120/80 mmHg",
                "heart_rate": "72 bpm",
                "temperature": "36.8 °C",
                "oxygen_saturation": "98%",
                "respiratory_rate": "16 breaths/min"
            },
            "assessment": {
                "general_condition": "Stable",
                "consciousness": "Alert and oriented",
                "cardiovascular": "Regular rate and rhythm",
                "respiratory": "Clear breath sounds",
                "gastrointestinal": "Normal bowel sounds"
            },
            "recommendations": [
                "Continue current medication regimen",
                "Follow up in 2 weeks",
                "Maintain healthy diet and exercise",
                "Report any new symptoms immediately"
            ],
            "next_steps": [
                "Schedule follow-up appointment",
                "Complete prescribed laboratory tests",
                "Review medication list at next visit"
            ]
        }
        
        return {
            "status": "success",
            "report": report,
            "generated_at": datetime.now().isoformat(),
            "doctor_name": current_user.name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 