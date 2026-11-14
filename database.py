"""
database.py
SQLAlchemy Database Models and Initializer:
- User (Authentication with password hashing)
- Patient (Clinical patient demographics and medical history)
- Analysis (Arrhythmia analysis results, statistics, and report paths)
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), default='Clinician')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patients = db.relationship('Patient', backref='clinician', lazy=True, cascade='all, delete-orphan')
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_id = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    contact = db.Column(db.String(50), nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    analyses = db.relationship('Analysis', backref='patient', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'full_name': self.full_name,
            'age': self.age,
            'gender': self.gender,
            'contact': self.contact or 'N/A',
            'medical_history': self.medical_history or 'None recorded',
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class Analysis(db.Model):
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20), default='CSV')
    sample_rate = db.Column(db.Float, default=360.0)
    duration_seconds = db.Column(db.Float, default=0.0)
    total_beats = db.Column(db.Integer, default=0)
    normal_beats = db.Column(db.Integer, default=0)
    abnormal_beats = db.Column(db.Integer, default=0)
    s_beats = db.Column(db.Integer, default=0)
    v_beats = db.Column(db.Integer, default=0)
    f_beats = db.Column(db.Integer, default=0)
    q_beats = db.Column(db.Integer, default=0)
    avg_heart_rate = db.Column(db.Float, default=0.0)
    overall_diagnosis = db.Column(db.String(50), default='Normal')
    confidence_score = db.Column(db.Float, default=0.0)
    algorithm = db.Column(db.String(100), default='Weighted K-Nearest Neighbors (WKNN)')
    ecg_plot_path = db.Column(db.String(255), nullable=True)
    pdf_report_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_name': self.patient.full_name if self.patient else 'Unknown',
            'patient_code': self.patient.patient_id if self.patient else 'Unknown',
            'filename': self.filename,
            'file_type': self.file_type,
            'duration_seconds': round(self.duration_seconds, 2),
            'total_beats': self.total_beats,
            'normal_beats': self.normal_beats,
            'abnormal_beats': self.abnormal_beats,
            's_beats': self.s_beats,
            'v_beats': self.v_beats,
            'f_beats': self.f_beats,
            'q_beats': self.q_beats,
            'avg_heart_rate': round(self.avg_heart_rate, 1),
            'overall_diagnosis': self.overall_diagnosis,
            'confidence_score': round(self.confidence_score * 100, 1),
            'algorithm': self.algorithm,
            'pdf_report_path': self.pdf_report_path,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

def init_db(app):
    """Initializes database and creates default clinician user if not present."""
    with app.app_context():
        db.create_all()
        # Seed default test clinician
        if not User.query.filter_by(username='doctor').first():
            demo_user = User(
                username='doctor',
                email='doctor@hospital.org',
                role='Lead Cardiologist'
            )
            demo_user.set_password('doctor123')
            db.session.add(demo_user)
            db.session.commit()
            print("Initialized database and created default clinician user: doctor / doctor123")
