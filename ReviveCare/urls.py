"""
URL configuration for ReviveCare project.
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from Home.views import *
from Doctor.views import *
from Patients.views import *


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name="home"),

    # Doctor Portal Authentication & Dashboard Routes
    path('doctor/', doc_port, name="doctor_dashboard"),
    path('doc_port/', doc_port, name="doc_port"),
    path('doctor/login/', doctor_login, name="doctor_login"),
    path('doctor/register/', doctor_register, name="doctor_register"),
    path('doctor/logout/', doctor_logout, name="doctor_logout"),

    # Doctor Patient Onboarding & PDF Extraction Routes
    path('doc_info_page/', doc_info_page, name="doc_info_page"),
    path('doctor/patient/add/', doc_info_page, name="doctor_add_patient"),
    path('doctor_info', doctor_info, name='doctor_info'),
    path('doctor/extract-pdf/', extract_pdf_api, name='extract_pdf_api'),

    # Doctor Patient Telemetry, Video Auto-Inference & Alert Review Routes
    path('doctor/patient/<int:patient_id>/', patient_detail, name='patient_detail'),
    path('doctor/patient/<int:patient_id>/analyze-video/', analyze_custom_exercise_video, name='analyze_custom_exercise_video'),
    path('doctor/patient/<int:patient_id>/prescribe/', prescribe_custom_exercise, name='prescribe_custom_exercise'),
    path('doctor/patient/<int:patient_id>/generate-summary/', doctor_generate_summary, name='doctor_generate_summary'),
    path('doctor/patient/<int:patient_id>/claim/', claim_unassigned_patient, name='claim_unassigned_patient'),
    path('doctor/alert/<int:alert_id>/review/', review_emergency_alert, name='review_emergency_alert'),
    path('doctor/custom-exercise/<int:exercise_id>/delete/', delete_custom_exercise, name='delete_custom_exercise'),

    # Patient Portal Routes
    path('login/', login, name='login'),
    path('patient/dashboard/', patient_dashboard, name='patient_dashboard'),
    path('patient/logout/', patient_logout, name='patient_logout'),
    path('patient/chatbot/', chatbot, name='chatbot'),
    path('patient/chatbot/send/', chatbot_send, name='chatbot_send'),
    path('exercise', exercise, name='exercise'),

    # Side Lateral Raise routes
    path('sr', sr, name='sr'),
    path('srtwo', srtwo, name='srtwo'),
    path('sr/video_feed/', video_feed, name='video_feed'),
    path('sr/start/', start_workout, name='start_workout'),
    path('sr/reset/', reset_workout, name='reset_workout'),
    path('sr/status/', get_workout_status, name='get_workout_status'),
    path('sr/save/', save_workout, name='save_workout'),

    # Bicep Curls live tracking routes
    path('bc', bc, name='bc'),
    path('bc/video_feed/', bc_video_feed, name='bc_video_feed'),
    path('bc/start/', start_bc_workout, name='start_bc_workout'),
    path('bc/reset/', reset_bc_workout, name='reset_bc_workout'),
    path('bc/status/', get_bc_status, name='get_bc_status'),
    path('bc/switch_arm/', switch_bc_arm, name='switch_bc_arm'),

    # Arm Raises live tracking routes
    path('ar', ar, name='ar'),
    path('ar/video_feed/', ar_video_feed, name='ar_video_feed'),
    path('ar/start/', start_ar_workout, name='start_ar_workout'),
    path('ar/reset/', reset_ar_workout, name='reset_ar_workout'),
    path('ar/status/', get_ar_status, name='get_ar_status'),

    # Modified Jumping Jacks live tracking routes
    path('jj', jj, name='jj'),
    path('jj/video_feed/', jj_video_feed, name='jj_video_feed'),
    path('jj/start/', start_jj_workout, name='start_jj_workout'),
    path('jj/reset/', reset_jj_workout, name='reset_jj_workout'),
    path('jj/status/', get_jj_status, name='get_jj_status'),

    # Summary generation
    path("generate-summary/", generate_and_send_summary, name="generate_and_send_summary"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
