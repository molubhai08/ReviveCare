from django.contrib import admin
from .models import Doctor, CustomExercise, EmergencyAlert


@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'specialty', 'hospital', 'created_at')
    search_fields = ('name', 'email', 'specialty', 'hospital')


@admin.register(CustomExercise)
class CustomExerciseAdmin(admin.ModelAdmin):
    list_display = ('name', 'patient', 'doctor', 'difficulty', 'target_sets', 'target_reps', 'created_at')
    list_filter = ('difficulty', 'doctor')
    search_fields = ('name', 'patient__name', 'doctor__name')


@admin.register(EmergencyAlert)
class EmergencyAlertAdmin(admin.ModelAdmin):
    list_display = ('patient', 'doctor', 'seriousness_score', 'is_reviewed', 'created_at')
    list_filter = ('is_reviewed', 'doctor')
    search_fields = ('patient__name', 'user_query')
