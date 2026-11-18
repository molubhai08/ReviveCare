from django.db import models
from django.utils import timezone


class Doctor(models.Model):
    """Doctor profile with email-based authentication"""
    name = models.CharField(max_length=150)
    email = models.EmailField(unique=True)
    specialty = models.CharField(max_length=150, blank=True, default='Orthopedic Specialist')
    hospital = models.CharField(max_length=200, blank=True, default='City General Hospital')
    phone = models.CharField(max_length=25, blank=True, default='')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Dr. {self.name} ({self.specialty})"

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Doctor'
        verbose_name_plural = 'Doctors'


class CustomExercise(models.Model):
    """Custom rehabilitation exercise prescribed by a doctor for an individual patient"""
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='custom_exercises')
    patient = models.ForeignKey('Patients.Patient', on_delete=models.CASCADE, related_name='custom_exercises')
    name = models.CharField(max_length=150)
    description = models.TextField(blank=True, default='')
    target_reps = models.IntegerField(default=10)
    target_sets = models.IntegerField(default=3)
    difficulty = models.CharField(max_length=50, default='Medium')
    video = models.FileField(upload_to='custom_exercises/', null=True, blank=True)
    kinematics_summary = models.TextField(blank=True, default='', help_text="AI/MediaPipe detected kinematic parameters")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} prescribed for {self.patient.name} by Dr. {self.doctor.name}"

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Custom Exercise'
        verbose_name_plural = 'Custom Exercises'


class EmergencyAlert(models.Model):
    """High-seriousness clinical alert triggered by patient queries in the chatbot"""
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True, blank=True, related_name='emergency_alerts')
    patient = models.ForeignKey('Patients.Patient', on_delete=models.CASCADE, related_name='emergency_alerts')
    user_query = models.TextField()
    ai_response = models.TextField()
    seriousness_score = models.FloatField(default=0.8)
    created_at = models.DateTimeField(default=timezone.now)
    is_reviewed = models.BooleanField(default=False)

    def __str__(self):
        return f"Alert: {self.patient.name} (Score: {self.seriousness_score:.2f}) - {'Reviewed' if self.is_reviewed else 'Pending'}"

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Emergency Alert'
        verbose_name_plural = 'Emergency Alerts'
