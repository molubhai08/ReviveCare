from django.contrib import admin
from .models import Patient, SLS, Logs


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'doctor', 'is_active', 'updated_at')
    list_filter = ('is_active', 'doctor')
    search_fields = ('name', 'email')


@admin.register(SLS)
class SLSAdmin(admin.ModelAdmin):
    list_display = ('patient', 'exercise_type', 'date', 'total_reps', 'target_reps', 'completed')
    list_filter = ('exercise_type', 'completed', 'date')
    search_fields = ('patient__name', 'exercise_type')


@admin.register(Logs)
class LogsAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'updated_at')
    search_fields = ('name', 'email')
