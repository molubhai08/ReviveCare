# patient/models.py
from django.db import models
from django.utils import timezone

class Patient(models.Model):
    doctor = models.ForeignKey('Doctor.Doctor', on_delete=models.SET_NULL, null=True, blank=True, related_name='patients')
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)   # 🔒 UNIQUE - This is the only authentication
    info = models.TextField(max_length=5000)
    summary = models.TextField(max_length=5000, null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'


# Add this model below Patient
class SLS(models.Model):
    """Side Lateral Raise & Rehabilitation Exercise workout tracking"""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='sls_workouts')
    exercise_type = models.CharField(max_length=50, default='Side Lateral Raise')
    date = models.DateField(default=timezone.now)
    
    # Workout metrics
    total_reps = models.IntegerField(default=0)
    target_reps = models.IntegerField(default=10)
    excellent_reps = models.IntegerField(default=0)
    good_reps = models.IntegerField(default=0)
    partial_reps = models.IntegerField(default=0)
    
    # Session details
    duration_seconds = models.IntegerField(default=0, help_text="Workout duration in seconds")
    completed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.patient.name} - {self.exercise_type} - {self.date}"
    
    @property
    def completion_percentage(self):
        """Calculate completion percentage"""
        if self.target_reps == 0:
            return 0
        return int((self.total_reps / self.target_reps) * 100)
    
    @property
    def quality_score(self):
        """Calculate quality score based on rep quality"""
        if self.total_reps == 0:
            return 0
        good_only = max(0, self.good_reps - self.excellent_reps) if (self.excellent_reps + self.good_reps + self.partial_reps > self.total_reps) else self.good_reps
        quality = (self.excellent_reps * 100 + good_only * 75 + self.partial_reps * 40) / self.total_reps
        return min(100.0, round(quality, 1))
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'SLS Workout'
        verbose_name_plural = 'SLS Workouts'


# models.py
class Logs(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='logs')
    name = models.CharField(max_length=1000)
    email = models.EmailField()
    text = models.TextField(max_length=100000, default="")  # Start empty, append to it
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} - Log"
    
    class Meta:
        verbose_name = 'Patient Log'
        verbose_name_plural = 'Patient Logs'
