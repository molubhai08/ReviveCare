import io
import json
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from Doctor.models import Doctor, CustomExercise, EmergencyAlert
from Patients.models import Patient, SLS, Logs


class ReViveCareIntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_doctor_register_login_logout_and_scoping(self):
        # 1. Register Doctor 1
        resp = self.client.post(reverse('doctor_register'), {
            'name': 'Dr. Ananya Sharma',
            'email': 'ananya@revivecare.org',
            'specialty': 'Orthopedic Rehab',
            'hospital': 'AIIMS Rehab Center',
        })
        self.assertEqual(resp.status_code, 302)
        doc1 = Doctor.objects.get(email='ananya@revivecare.org')
        self.assertEqual(self.client.session.get('doctor_id'), doc1.id)

        # Create a patient for Doctor 1 and another for Doctor 2
        doc2 = Doctor.objects.create(name='Vikram Rao', email='vikram@revivecare.org')
        p1 = Patient.objects.create(name='Rohan Gupta', email='rohan@example.com', info='ACL reconstruction post-op', doctor=doc1)
        p2 = Patient.objects.create(name='Meera Nair', email='meera@example.com', info='Rotator cuff repair', doctor=doc2)

        # Dashboard should show only Doctor 1's patient
        dash_resp = self.client.get(reverse('doc_port'))
        self.assertEqual(dash_resp.status_code, 200)
        self.assertContains(dash_resp, 'Rohan Gupta')
        self.assertNotContains(dash_resp, 'Meera Nair')

        # Doctor 1 cannot access Doctor 2's patient detail page (404 enforced by doctor_id scoping)
        detail_p1 = self.client.get(reverse('patient_detail', args=[p1.id]))
        self.assertEqual(detail_p1.status_code, 200)
        detail_p2 = self.client.get(reverse('patient_detail', args=[p2.id]))
        self.assertEqual(detail_p2.status_code, 404)

        # Logout and Login with email only
        self.client.get(reverse('doctor_logout'))
        self.assertIsNone(self.client.session.get('doctor_id'))

        login_resp = self.client.post(reverse('doctor_login'), {'email': 'ananya@revivecare.org'})
        self.assertEqual(login_resp.status_code, 302)
        self.assertEqual(self.client.session.get('doctor_id'), doc1.id)

    @patch('Doctor.views.extract_text_from_pdf')
    def test_patient_onboarding_via_pdf_extraction(self, mock_extract):
        mock_extract.return_value = ("DISCHARGE SUMMARY: Total Knee Arthroplasty (Right). Weight bearing as tolerated.", 2)
        doc = Doctor.objects.create(name='Aarav Mehta', email='aarav@revivecare.org')
        session = self.client.session
        session['doctor_id'] = doc.id
        session.save()

        fake_pdf = SimpleUploadedFile("discharge.pdf", b"%PDF-1.4 fake pdf bytes", content_type="application/pdf")
        resp = self.client.post(reverse('doctor_info'), {
            'name': 'Kabir Bedi',
            'email': 'kabir@example.com',
            'info': 'Follow-up in 2 weeks.',
            'pdf_file': fake_pdf,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])

        patient = Patient.objects.get(email='kabir@example.com')
        self.assertEqual(patient.doctor, doc)
        self.assertIn("Total Knee Arthroplasty", patient.info)
        self.assertIn("Follow-up in 2 weeks.", patient.info)

    def test_custom_exercise_prescription_and_patient_display(self):
        doc = Doctor.objects.create(name='Aarav Mehta', email='aarav@revivecare.org')
        patient = Patient.objects.create(name='Priya Verma', email='priya@example.com', info='Shoulder impingement', doctor=doc)

        session = self.client.session
        session['doctor_id'] = doc.id
        session['patient_id'] = patient.id
        session.save()

        fake_video = SimpleUploadedFile("wall_slides.mp4", b"\x00\x00\x00\x18ftypmp42", content_type="video/mp4")
        resp = self.client.post(
            reverse('prescribe_custom_exercise', args=[patient.id]),
            {
                'name': 'Scapular Wall Slides',
                'description': 'Slide forearms smoothly up the wall keeping scapulae retracted.',
                'target_reps': 12,
                'target_sets': 3,
                'difficulty': 'Medium',
                'kinematics_summary': 'Primary Active Joint: Shoulders (Peak ROM: 112.5°)',
                'video': fake_video,
            },
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(CustomExercise.objects.filter(patient=patient, doctor=doc).count(), 1)

        # Verify Patient Exercise Portal renders both Custom Exercise and 4 Default Exercises
        ex_resp = self.client.get(reverse('exercise'))
        self.assertEqual(ex_resp.status_code, 200)
        self.assertContains(ex_resp, 'Prescribed by Your Doctor')
        self.assertContains(ex_resp, 'Scapular Wall Slides')
        self.assertContains(ex_resp, 'Bicep Curls')
        self.assertContains(ex_resp, 'Side Lateral Raises')
        self.assertContains(ex_resp, 'Modified Jumping Jacks')
        self.assertContains(ex_resp, 'Overhead Arm Raises')

    @patch('Patients.views.send_email_alert')
    @patch('Patients.views.initialize_model')
    def test_chatbot_high_seriousness_creates_emergency_alert(self, mock_init_model, mock_email):
        doc = Doctor.objects.create(name='Aarav Mehta', email='aarav@revivecare.org')
        patient = Patient.objects.create(name='Priya Verma', email='priya@example.com', info='Post-op knee surgery', doctor=doc)

        mock_llm = MagicMock()
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = json.dumps({
            "patient_response": "Please contact Dr. Aarav Mehta or visit the emergency department immediately.",
            "seriousness_score": 0.88
        })
        mock_llm.invoke.return_value = mock_ai_msg
        mock_init_model.return_value = mock_llm

        session = self.client.session
        session['patient_id'] = patient.id
        session['doctor_id'] = doc.id
        session.save()

        resp = self.client.post(
            reverse('chatbot_send'),
            data=json.dumps({'message': 'My calf is very swollen, hot, and painful with chest shortness of breath'}),
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 200)
        alert = EmergencyAlert.objects.filter(patient=patient, doctor=doc).first()
        self.assertIsNotNone(alert)
        self.assertAlmostEqual(alert.seriousness_score, 0.88)
        self.assertFalse(alert.is_reviewed)

        # Mark alert reviewed from Doctor Portal
        rev_resp = self.client.post(
            reverse('review_emergency_alert', args=[alert.id]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(rev_resp.status_code, 200)
        alert.refresh_from_db()
        self.assertTrue(alert.is_reviewed)
