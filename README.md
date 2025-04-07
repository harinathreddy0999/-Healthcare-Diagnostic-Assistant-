
# 🩺 # Healthcare Diagnostic Assistant

A modern web application that combines healthcare management with AI-powered diagnostic capabilities. This application provides a comprehensive suite of tools for healthcare professionals to manage patients, appointments, and leverage AI for medical analysis.

![Healthcare Diagnostic Assistant Dashboard](./docs/dashboard.png)

## Features

### 1. User Management
- Secure authentication system with JWT tokens
- User registration and login
- Role-based access control
- Profile management

### 2. Dashboard
- Overview of key metrics
- Total patients count
- Total appointments
- Recent patient list
- Quick access to all major features

### 3. AI Services

#### Medical Image Analysis
- AI-powered analysis of medical images
- Support for X-rays, MRIs, and CT scans
- Confidence scores for findings
- Detailed image statistics

#### Diagnostic Assistant
- Symptom analysis
- Preliminary diagnosis suggestions
- AI-driven medical recommendations
- Confidence levels for diagnoses

#### Patient Monitoring
- Real-time vital signs monitoring
- Automated alerts for abnormal readings
- Historical trend analysis
- Patient status tracking

#### Medical Documentation
- Automated report generation
- Clinical notes management
- Digital record keeping
- Structured medical documentation

## Technology Stack

### Frontend
- Next.js 15.2.4
- React
- TypeScript
- Tailwind CSS
- JWT Authentication

### Backend
- FastAPI
- SQLite Database
- Python 3.12
- JWT Authentication
- AI/ML Libraries (TensorFlow, NumPy, scikit-learn)

## Getting Started

### Prerequisites
- Node.js (v18 or higher)
- Python 3.12
- npm or yarn
- Virtual environment for Python

### Installation

1. Clone the repository
```bash
git clone [repository-url]
cd Healthcare-Diagnostic-Assistant
```

2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

4. Access the application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Backend (.env)
```
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///./healthcare.db
```

## API Endpoints

### Authentication
- POST /auth/register - User registration
- POST /auth/login - User login
- GET /users/me - Get current user

### Patients
- GET /patients - List all patients
- POST /patients - Create new patient
- GET /patients/{id} - Get patient details
- PUT /patients/{id} - Update patient
- DELETE /patients/{id} - Delete patient

### AI Services
- POST /services/analyze-image - Medical image analysis
- POST /services/diagnose - Symptom analysis
- POST /services/monitor - Patient monitoring
- POST /services/generate-report - Report generation

## Security

- JWT token-based authentication
- Password hashing with bcrypt
- CORS protection
- Environment variable configuration
- Secure cookie handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical image analysis powered by TensorFlow
- Diagnostic algorithms based on latest medical research
- UI/UX design inspired by modern healthcare applications 

UI inspired by modern healthcare systems
