import io
import json
import os
import re
import tempfile
from datetime import date, timedelta

import cv2
import mediapipe as mp
import numpy as np
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods, require_POST
from dotenv import load_dotenv
from pypdf import PdfReader

from Doctor.models import CustomExercise, Doctor, EmergencyAlert
from Patients.models import Logs, Patient, SLS

load_dotenv()

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ============================================================================
# HELPER UTILITIES (AUTH, KINEMATICS, GROQ AI, PDF EXTRACTION)
# ============================================================================

def get_logged_in_doctor(request):
    """Retrieve currently authenticated doctor from session"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return None
    return Doctor.objects.filter(id=doctor_id).first()


def get_groq_llm(temperature=0.25, max_tokens=1024):
    """Initialize ChatGroq using GROQ_MODEL from .env"""
    if not LANGCHAIN_AVAILABLE:
        return None
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
        max_retries=2,
    )


def extract_text_from_pdf(pdf_file):
    """Extract plain clinical text from an uploaded PDF file using pypdf.PdfReader"""
    try:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        extracted_pages = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                extracted_pages.append(page_text)
        full_text = "\n\n".join(extracted_pages).strip()
        # Normalize excessive blank lines
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        return full_text, len(reader.pages)
    except Exception as e:
        raise ValueError(f"Unable to read PDF file: {str(e)}")


def _calc_angle_2d(a, b, c):
    """Calculate 2D joint angle in degrees between points a-b-c (b is vertex)"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def analyze_exercise_video_kinematics(video_path):
    """
    Process an uploaded exercise demo video with OpenCV and MediaPipe Pose.
    Samples frames across the video to track movement across primary joints:
    shoulders, elbows, hips, and knees, calculating peak angular displacement (ROM).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "kinematics_summary": "Video uploaded (unable to decode frame stream; manual clinical parameters enabled).",
            "joint_metrics": {},
            "frames_analyzed": 0,
            "primary_joint": "Shoulders",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0

    # Sample up to 48 frames evenly across the video (or every 3rd frame if total_frames is unknown/WebM)
    max_samples = 48
    if total_frames > max_samples:
        step = max(1, total_frames // max_samples)
    else:
        step = 2

    joint_history = {
        "left_shoulder": [],
        "right_shoulder": [],
        "left_elbow": [],
        "right_elbow": [],
        "left_hip": [],
        "right_hip": [],
        "left_knee": [],
        "right_knee": [],
    }
    wrist_vertical_range = []
    ankle_lateral_range = []

    mp_pose = mp.solutions.pose
    frames_read = 0
    frames_with_pose = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as pose:
        while cap.isOpened() and frames_with_pose < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            if frames_read % step != 0 and frames_read != 1:
                continue

            # Resize large frames for fast MediaPipe inference
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_landmarks:
                continue

            frames_with_pose += 1
            lm = results.pose_landmarks.landmark
            PL = mp_pose.PoseLandmark

            def pt(idx):
                return [lm[idx.value].x, lm[idx.value].y]

            def vis(idx):
                return lm[idx.value].visibility

            # Shoulders: Hip - Shoulder - Elbow
            if vis(PL.LEFT_HIP) > 0.3 and vis(PL.LEFT_SHOULDER) > 0.3 and vis(PL.LEFT_ELBOW) > 0.3:
                joint_history["left_shoulder"].append(
                    _calc_angle_2d(pt(PL.LEFT_HIP), pt(PL.LEFT_SHOULDER), pt(PL.LEFT_ELBOW))
                )
            if vis(PL.RIGHT_HIP) > 0.3 and vis(PL.RIGHT_SHOULDER) > 0.3 and vis(PL.RIGHT_ELBOW) > 0.3:
                joint_history["right_shoulder"].append(
                    _calc_angle_2d(pt(PL.RIGHT_HIP), pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_ELBOW))
                )

            # Elbows: Shoulder - Elbow - Wrist
            if vis(PL.LEFT_SHOULDER) > 0.3 and vis(PL.LEFT_ELBOW) > 0.3 and vis(PL.LEFT_WRIST) > 0.3:
                joint_history["left_elbow"].append(
                    _calc_angle_2d(pt(PL.LEFT_SHOULDER), pt(PL.LEFT_ELBOW), pt(PL.LEFT_WRIST))
                )
                wrist_vertical_range.append(lm[PL.LEFT_WRIST.value].y)
            if vis(PL.RIGHT_SHOULDER) > 0.3 and vis(PL.RIGHT_ELBOW) > 0.3 and vis(PL.RIGHT_WRIST) > 0.3:
                joint_history["right_elbow"].append(
                    _calc_angle_2d(pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_ELBOW), pt(PL.RIGHT_WRIST))
                )
                wrist_vertical_range.append(lm[PL.RIGHT_WRIST.value].y)

            # Hips: Shoulder - Hip - Knee
            if vis(PL.LEFT_SHOULDER) > 0.3 and vis(PL.LEFT_HIP) > 0.3 and vis(PL.LEFT_KNEE) > 0.3:
                joint_history["left_hip"].append(
                    _calc_angle_2d(pt(PL.LEFT_SHOULDER), pt(PL.LEFT_HIP), pt(PL.LEFT_KNEE))
                )
            if vis(PL.RIGHT_SHOULDER) > 0.3 and vis(PL.RIGHT_HIP) > 0.3 and vis(PL.RIGHT_KNEE) > 0.3:
                joint_history["right_hip"].append(
                    _calc_angle_2d(pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_HIP), pt(PL.RIGHT_KNEE))
                )

            # Knees: Hip - Knee - Ankle
            if vis(PL.LEFT_HIP) > 0.3 and vis(PL.LEFT_KNEE) > 0.3 and vis(PL.LEFT_ANKLE) > 0.3:
                joint_history["left_knee"].append(
                    _calc_angle_2d(pt(PL.LEFT_HIP), pt(PL.LEFT_KNEE), pt(PL.LEFT_ANKLE))
                )
            if vis(PL.RIGHT_HIP) > 0.3 and vis(PL.RIGHT_KNEE) > 0.3 and vis(PL.RIGHT_ANKLE) > 0.3:
                joint_history["right_knee"].append(
                    _calc_angle_2d(pt(PL.RIGHT_HIP), pt(PL.RIGHT_KNEE), pt(PL.RIGHT_ANKLE))
                )
            if vis(PL.LEFT_ANKLE) > 0.3 and vis(PL.RIGHT_ANKLE) > 0.3:
                ankle_lateral_range.append(abs(lm[PL.LEFT_ANKLE.value].x - lm[PL.RIGHT_ANKLE.value].x))

    cap.release()

    def _summarize_joint(left_vals, right_vals):
        l_rom = (max(left_vals) - min(left_vals)) if len(left_vals) >= 2 else 0.0
        r_rom = (max(right_vals) - min(right_vals)) if len(right_vals) >= 2 else 0.0
        peak_rom = max(l_rom, r_rom)
        all_vals = left_vals + right_vals
        min_deg = min(all_vals) if all_vals else 0.0
        max_deg = max(all_vals) if all_vals else 0.0
        symmetry_diff = abs(l_rom - r_rom)
        return {
            "left_rom": round(l_rom, 1),
            "right_rom": round(r_rom, 1),
            "peak_rom": round(peak_rom, 1),
            "min_angle": round(min_deg, 1),
            "max_angle": round(max_deg, 1),
            "bilateral": bool(l_rom > 18 and r_rom > 18 and symmetry_diff < max(25.0, peak_rom * 0.45)),
        }

    joint_metrics = {
        "Shoulders": _summarize_joint(joint_history["left_shoulder"], joint_history["right_shoulder"]),
        "Elbows": _summarize_joint(joint_history["left_elbow"], joint_history["right_elbow"]),
        "Hips": _summarize_joint(joint_history["left_hip"], joint_history["right_hip"]),
        "Knees": _summarize_joint(joint_history["left_knee"], joint_history["right_knee"]),
    }

    sorted_joints = sorted(joint_metrics.items(), key=lambda kv: kv[1]["peak_rom"], reverse=True)
    primary_joint = sorted_joints[0][0] if sorted_joints else "Shoulders"
    secondary_joint = sorted_joints[1][0] if len(sorted_joints) > 1 else "Elbows"

    est_duration = round(frames_read / fps, 1) if frames_read > 0 else 0.0
    laterality = "Bilateral" if joint_metrics[primary_joint]["bilateral"] else "Unilateral / Alternating"

    summary_parts = [
        f"Pose Frames Tracked: {frames_with_pose} (Duration: ~{est_duration}s)",
        f"Primary Active Joint: {primary_joint} (Peak ROM: {joint_metrics[primary_joint]['peak_rom']}°, "
        f"Range: {joint_metrics[primary_joint]['min_angle']}°–{joint_metrics[primary_joint]['max_angle']}°)",
        f"Secondary Joint: {secondary_joint} (Peak ROM: {joint_metrics[secondary_joint]['peak_rom']}°)",
        f"All Joint ROMs — Shoulders: {joint_metrics['Shoulders']['peak_rom']}° "
        f"(L:{joint_metrics['Shoulders']['left_rom']}°/R:{joint_metrics['Shoulders']['right_rom']}°), "
        f"Elbows: {joint_metrics['Elbows']['peak_rom']}° "
        f"(L:{joint_metrics['Elbows']['left_rom']}°/R:{joint_metrics['Elbows']['right_rom']}°), "
        f"Hips: {joint_metrics['Hips']['peak_rom']}° "
        f"(L:{joint_metrics['Hips']['left_rom']}°/R:{joint_metrics['Hips']['right_rom']}°), "
        f"Knees: {joint_metrics['Knees']['peak_rom']}° "
        f"(L:{joint_metrics['Knees']['left_rom']}°/R:{joint_metrics['Knees']['right_rom']}°)",
        f"Movement Pattern: {laterality} {primary_joint} & {secondary_joint} Mobilization",
    ]
    kinematics_summary = " | ".join(summary_parts)

    return {
        "kinematics_summary": kinematics_summary,
        "joint_metrics": joint_metrics,
        "frames_analyzed": frames_with_pose,
        "primary_joint": primary_joint,
        "secondary_joint": secondary_joint,
        "laterality": laterality,
    }


def _heuristic_exercise_fallback(kinematics_data):
    """Fallback clinical exercise inference if Groq API is unavailable"""
    primary = kinematics_data.get("primary_joint", "Shoulders")
    joint_metrics = kinematics_data.get("joint_metrics", {})
    rom = joint_metrics.get(primary, {}).get("peak_rom", 60.0)

    if primary == "Knees":
        return {
            "name": "Seated Knee Extensions",
            "description": (
                f"Controlled knee extension movement targeting quadriceps activation and joint mobility "
                f"(detected knee ROM ~{rom}°). Sit upright with back supported, slowly extend the knee to "
                f"peak extension, hold for 2 seconds, and lower with a 3-second eccentric phase."
            ),
            "target_reps": 12,
            "target_sets": 3,
            "difficulty": "Medium" if rom >= 55 else "Easy",
        }
    elif primary == "Hips":
        return {
            "name": "Supported Hip Flexion & Sit-to-Stand",
            "description": (
                f"Progressive hip mobility and lower-extremity strengthening exercise (detected hip ROM ~{rom}°). "
                f"Maintain neutral spine alignment, engage core stabilizers, and move smoothly through the pain-free range."
            ),
            "target_reps": 10,
            "target_sets": 3,
            "difficulty": "Medium",
        }
    elif primary == "Elbows":
        return {
            "name": "Controlled Elbow Flexion & Wall Slides",
            "description": (
                f"Upper-limb elbow and forearm rehabilitation protocol (detected elbow ROM ~{rom}°). "
                f"Keep shoulders stabilized down and back while flexing and extending through a smooth, controlled arc."
            ),
            "target_reps": 12,
            "target_sets": 3,
            "difficulty": "Easy" if rom < 70 else "Medium",
        }
    else:
        return {
            "name": "Scapular Wall Slides & Shoulder Flexion",
            "description": (
                f"Guided shoulder girdle mobility and rotator cuff stabilization exercise (detected shoulder ROM ~{rom}°). "
                f"Elevate arms smoothly without shrugging the upper trapezius, pause briefly at peak range, and return slowly."
            ),
            "target_reps": 10,
            "target_sets": 3,
            "difficulty": "Medium" if rom >= 75 else "Easy",
        }


def infer_exercise_with_groq(kinematics_data, patient=None, filename=""):
    """
    Feed the MediaPipe kinematic summary to Groq (qwen/qwen3.8-27b) to auto-infer:
    - Suggested Exercise Name
    - Clinical Description & Execution Tips
    - Target Reps, Sets, and Difficulty
    """
    fallback = _heuristic_exercise_fallback(kinematics_data)
    llm = get_groq_llm(temperature=0.2, max_tokens=700)
    if not llm:
        return fallback

    patient_context = ""
    if patient and patient.info:
        patient_context = f"\nPATIENT CLINICAL CONTEXT:\n{patient.info[:1200]}\n"

    system_prompt = """You are an expert orthopedic physiotherapist and biomechanics AI assistant for ReViveCare.
Given MediaPipe Pose kinematic telemetry extracted from a doctor's exercise demonstration video (including peak angular displacement / Range of Motion across Shoulders, Elbows, Hips, and Knees), infer the rehabilitation exercise details.

Return ONLY a valid JSON object with these exact keys:
{
  "name": "Concise Clinical Exercise Name (e.g., Scapular Wall Slides, Seated Knee Extensions, Standing Hip Abduction)",
  "description": "2-3 sentences of clinical description, biomechanical purpose, posture execution tips, and safety cues referencing the detected ROM.",
  "target_reps": 10,
  "target_sets": 3,
  "difficulty": "Easy"
}
Rules:
- difficulty MUST be one of: "Easy", "Medium", "Hard"
- target_reps must be an integer between 5 and 20
- target_sets must be an integer between 1 and 5
- Output valid JSON only, with no markdown fences or extra commentary."""

    human_prompt = f"""VIDEO FILENAME HINT: {filename or 'demo_exercise.mp4'}
MEDIAPIPE POSE KINEMATIC TELEMETRY:
{kinematics_data.get('kinematics_summary', '')}
{patient_context}
Generate the JSON prescription parameters now:"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        raw_text = response.content.strip()
        # Strip <think>...</think> blocks if model emits reasoning tags
        raw_text = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            parsed = json.loads(json_match.group(0))
            difficulty = str(parsed.get("difficulty", fallback["difficulty"])).capitalize()
            if difficulty not in ("Easy", "Medium", "Hard"):
                difficulty = "Medium"
            return {
                "name": str(parsed.get("name") or fallback["name"]).strip()[:150],
                "description": str(parsed.get("description") or fallback["description"]).strip(),
                "target_reps": max(1, min(30, int(parsed.get("target_reps", fallback["target_reps"])))),
                "target_sets": max(1, min(10, int(parsed.get("target_sets", fallback["target_sets"])))),
                "difficulty": difficulty,
            }
    except Exception as e:
        print(f"Groq exercise inference fallback used: {e}")

    return fallback


def compute_patient_telemetry(patient, doctor_id=None):
    """Compute exercise streak, compliance rate, average quality score, and recent workouts for a patient"""
    workouts = list(SLS.objects.filter(patient=patient).order_by('-date', '-created_at'))
    total_workouts = len(workouts)

    if total_workouts > 0:
        avg_compliance = int(round(
            sum(min(100, w.completion_percentage) for w in workouts) / total_workouts
        ))
        avg_quality = round(
            sum(w.quality_score for w in workouts) / total_workouts, 1
        )
        total_reps = sum(w.total_reps for w in workouts)
        total_minutes = max(1, int(round(sum(w.duration_seconds for w in workouts) / 60.0)))

        # Calculate consecutive day streak from unique workout dates
        unique_dates = sorted({w.date for w in workouts}, reverse=True)
        streak = 1
        for i in range(len(unique_dates) - 1):
            if (unique_dates[i] - unique_dates[i + 1]).days == 1:
                streak += 1
            else:
                break
        last_workout_date = unique_dates[0]
    else:
        avg_compliance = 0
        avg_quality = 0.0
        total_reps = 0
        total_minutes = 0
        streak = 0
        last_workout_date = None

    custom_count = CustomExercise.objects.filter(patient=patient).count()
    alert_qs = EmergencyAlert.objects.filter(patient=patient, is_reviewed=False)
    if doctor_id:
        alert_qs = alert_qs.filter(doctor_id=doctor_id)
    pending_alert_count = alert_qs.count()

    return {
        "patient": patient,
        "total_workouts": total_workouts,
        "compliance_rate": avg_compliance,
        "avg_quality": avg_quality,
        "total_reps": total_reps,
        "total_minutes": total_minutes,
        "streak": streak,
        "last_workout_date": last_workout_date,
        "custom_count": custom_count,
        "pending_alert_count": pending_alert_count,
        "workouts": workouts,
    }


def parse_chat_logs(log_text):
    """Parse structured chat & workout entries from Logs.text for display in Patient Detail"""
    if not log_text:
        return []
    blocks = [b.strip() for b in re.split(r'={20,}', log_text) if b.strip()]
    parsed_entries = []
    for block in reversed(blocks):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timestamp = ""
        content_lines = []
        for line in lines:
            ts_match = re.match(r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]$', line)
            if ts_match and not timestamp:
                timestamp = ts_match.group(1)
            else:
                content_lines.append(line)
        body = "\n".join(content_lines).strip()
        if not body:
            continue
        entry_type = "system"
        if body.startswith("Patient :"):
            entry_type = "patient"
            body = body.replace("Patient :", "", 1).strip()
        elif body.startswith("AI  :") or body.startswith("AI :"):
            entry_type = "ai"
            body = re.sub(r'^AI\s*:\s*', '', body).strip()
        elif "WORKOUT" in body:
            entry_type = "workout"
        parsed_entries.append({
            "timestamp": timestamp,
            "type": entry_type,
            "text": body,
        })
    return parsed_entries[:40]


# ============================================================================
# A. DOCTOR AUTHENTICATION (EMAIL-ONLY)
# ============================================================================

def doctor_login(request):
    """Email-only login for doctors"""
    existing_doc = get_logged_in_doctor(request)
    if existing_doc and request.method == 'GET':
        return redirect('doc_port')

    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        if not email:
            messages.error(request, 'Please enter your email address.')
            return render(request, 'doctor_login.html', {'mode': 'login'})

        doctor = Doctor.objects.filter(email__iexact=email).first()
        if doctor:
            request.session['doctor_id'] = doctor.id
            request.session['doctor_name'] = doctor.name
            request.session['doctor_email'] = doctor.email
            messages.success(request, f'Welcome back, Dr. {doctor.name}!')
            return redirect('doc_port')
        else:
            messages.error(
                request,
                f'No doctor account found for "{email}". Please register below to create your clinical profile.'
            )
            return render(request, 'doctor_login.html', {
                'mode': 'register',
                'prefill_email': email,
            })

    return render(request, 'doctor_login.html', {'mode': 'login'})


def doctor_register(request):
    """Email-only registration for doctors (Name, Email, Specialty, Hospital)"""
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip().lower()
        specialty = request.POST.get('specialty', '').strip() or 'Orthopedic & Rehab Specialist'
        hospital = request.POST.get('hospital', '').strip() or 'ReViveCare Medical Center'
        phone = request.POST.get('phone', '').strip()

        if not name or not email:
            messages.error(request, 'Both Name and Email are required to register.')
            return render(request, 'doctor_login.html', {
                'mode': 'register',
                'prefill_name': name,
                'prefill_email': email,
                'prefill_specialty': specialty,
                'prefill_hospital': hospital,
            })

        # Clean "Dr." prefix if the user typed it so we don't double-prefix "Dr. Dr."
        clean_name = re.sub(r'^(dr\.?\s+)', '', name, flags=re.IGNORECASE).strip() or name

        doctor, created = Doctor.objects.update_or_create(
            email=email,
            defaults={
                'name': clean_name,
                'specialty': specialty,
                'hospital': hospital,
                'phone': phone,
            }
        )

        request.session['doctor_id'] = doctor.id
        request.session['doctor_name'] = doctor.name
        request.session['doctor_email'] = doctor.email

        if created:
            messages.success(request, f'Account registered! Welcome to ReViveCare, Dr. {doctor.name}.')
        else:
            messages.success(request, f'Welcome back, Dr. {doctor.name}! Your profile has been updated.')
        return redirect('doc_port')

    return render(request, 'doctor_login.html', {'mode': 'register'})


def doctor_logout(request):
    """Clear doctor session keys"""
    for key in ('doctor_id', 'doctor_name', 'doctor_email'):
        request.session.pop(key, None)
    messages.success(request, 'You have been logged out of the Doctor Portal.')
    return redirect('doctor_login')


# ============================================================================
# B. DOCTOR DASHBOARD & PATIENT TELEMETRY PORTAL
# ============================================================================

def doc_port(request):
    """
    Doctor Dashboard scoped strictly by doctor_id = request.session.get('doctor_id').
    Displays doctor profile, assigned patients telemetry, real-time emergency alerts,
    and quick onboarding actions.
    """
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return redirect('doctor_login')

    doctor = Doctor.objects.filter(id=doctor_id).first()
    if not doctor:
        request.session.pop('doctor_id', None)
        return redirect('doctor_login')

    # Strict data scoping by doctor_id
    assigned_patients = Patient.objects.filter(doctor_id=doctor_id).order_by('-updated_at')
    pending_alerts = EmergencyAlert.objects.filter(
        doctor_id=doctor_id,
        is_reviewed=False
    ).select_related('patient').order_by('-created_at')

    patient_rows = [compute_patient_telemetry(p, doctor_id=doctor_id) for p in assigned_patients]

    total_patients = len(patient_rows)
    pending_alerts_count = pending_alerts.count()
    custom_exercises_count = CustomExercise.objects.filter(doctor_id=doctor_id).count()
    active_rehab_plans = sum(1 for row in patient_rows if row["patient"].is_active)

    if total_patients > 0:
        overall_compliance = int(round(
            sum(r["compliance_rate"] for r in patient_rows) / total_patients
        ))
    else:
        overall_compliance = 0

    # Check if there are unassigned legacy patients in DB that the doctor can claim with 1 click
    unassigned_patients = Patient.objects.filter(doctor__isnull=True).order_by('-updated_at')

    # Compute initials for avatar
    initials = "".join([part[0].upper() for part in doctor.name.split()[:2]]) or "DR"

    context = {
        'doctor': doctor,
        'doctor_initials': initials,
        'patient_rows': patient_rows,
        'total_patients': total_patients,
        'pending_alerts': pending_alerts,
        'pending_alerts_count': pending_alerts_count,
        'active_rehab_plans': active_rehab_plans,
        'custom_exercises_count': custom_exercises_count,
        'overall_compliance': overall_compliance,
        'unassigned_patients': unassigned_patients,
    }
    return render(request, 'doc_port.html', context)


@require_POST
def claim_unassigned_patient(request, patient_id):
    """Assign an existing unassigned patient record (and any unassigned alerts) to the logged-in doctor"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return redirect('doctor_login')

    doctor = get_object_or_404(Doctor, id=doctor_id)
    patient = get_object_or_404(Patient, id=patient_id)
    if patient.doctor_id is None or patient.doctor_id == doctor.id:
        patient.doctor = doctor
        patient.save(update_fields=['doctor', 'updated_at'])
        EmergencyAlert.objects.filter(patient=patient, doctor__isnull=True).update(doctor=doctor)
        messages.success(request, f'{patient.name} has been assigned to your clinical care roster.')
    return redirect('doc_port')


@require_POST
def review_emergency_alert(request, alert_id):
    """Mark an EmergencyAlert as reviewed (scoped strictly to logged-in doctor)"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)

    alert = get_object_or_404(EmergencyAlert, id=alert_id, doctor_id=doctor_id)
    alert.is_reviewed = True
    alert.save(update_fields=['is_reviewed'])

    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', ''):
        return JsonResponse({'success': True, 'alert_id': alert.id})

    messages.success(request, f'Emergency alert for {alert.patient.name} marked as reviewed.')
    next_url = request.POST.get('next') or 'doc_port'
    return redirect(next_url)


# ============================================================================
# C. PATIENT ONBOARDING VIA PDF EXTRACTION
# ============================================================================

@ensure_csrf_cookie
def doc_info_page(request):
    """Render patient onboarding / clinical context update page (scoped to logged-in doctor)"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return redirect('doctor_login')

    doctor = get_object_or_404(Doctor, id=doctor_id)
    edit_patient = None
    patient_id = request.GET.get('patient_id')
    if patient_id:
        edit_patient = Patient.objects.filter(id=patient_id, doctor_id=doctor_id).first()

    return render(request, 'add_info.html', {
        'doctor': doctor,
        'edit_patient': edit_patient,
    })


@require_POST
def extract_pdf_api(request):
    """AJAX endpoint to extract plain text from an uploaded medical PDF via pypdf.PdfReader"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return JsonResponse({'success': False, 'error': 'Doctor authentication required.'}, status=401)

    pdf_file = request.FILES.get('pdf_file')
    if not pdf_file:
        return JsonResponse({'success': False, 'error': 'No PDF file uploaded.'}, status=400)

    if not pdf_file.name.lower().endswith('.pdf'):
        return JsonResponse({'success': False, 'error': 'Please upload a valid .pdf document.'}, status=400)

    try:
        extracted_text, page_count = extract_text_from_pdf(pdf_file)
        if not extracted_text:
            return JsonResponse({
                'success': False,
                'error': 'The uploaded PDF appears to contain scanned images without selectable text.'
            }, status=400)

        return JsonResponse({
            'success': True,
            'extracted_text': extracted_text,
            'pages': page_count,
            'filename': pdf_file.name,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@require_POST
def doctor_info(request):
    """
    Create or update a Patient assigned to the logged-in doctor.
    Supports both multipart/form-data (with optional Medical PDF upload extracted via pypdf.PdfReader)
    and JSON payloads.
    """
    doctor_id = request.session.get('doctor_id')
    doctor = Doctor.objects.filter(id=doctor_id).first() if doctor_id else Doctor.objects.first()

    content_type = request.content_type or ''
    pdf_extracted_text = ''

    if 'multipart/form-data' in content_type or request.FILES or request.POST:
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip().lower()
        manual_info = request.POST.get('info', '').strip()
        pdf_file = request.FILES.get('pdf_file')

        if pdf_file:
            try:
                pdf_extracted_text, _ = extract_text_from_pdf(pdf_file)
            except Exception as e:
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
    else:
        try:
            data = json.loads(request.body)
        except Exception:
            return JsonResponse({'success': False, 'message': 'Invalid request payload.'}, status=400)
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        manual_info = data.get('info', '').strip()

    if not name or not email:
        return JsonResponse({'success': False, 'message': 'Patient name and email are required.'}, status=400)

    # Combine manual clinical notes and extracted PDF text cleanly without duplication
    if pdf_extracted_text and manual_info:
        if pdf_extracted_text in manual_info:
            final_info = manual_info
        else:
            final_info = f"{manual_info}\n\n--- EXTRACTED MEDICAL PDF REPORT ---\n{pdf_extracted_text}"
    elif pdf_extracted_text:
        final_info = pdf_extracted_text
    else:
        final_info = manual_info

    if not final_info:
        return JsonResponse({
            'success': False,
            'message': 'Please provide clinical notes or upload a medical PDF report.'
        }, status=400)

    # Check if patient exists (case-insensitive email lookup)
    existing_patient = Patient.objects.filter(email__iexact=email).first()
    if existing_patient:
        existing_patient.name = name
        existing_patient.info = final_info
        if doctor:
            existing_patient.doctor = doctor
        existing_patient.save()
        patient = existing_patient
        created = False
    else:
        patient = Patient.objects.create(
            name=name,
            email=email,
            info=final_info,
            doctor=doctor,
        )
        created = True

    return JsonResponse({
        'success': True,
        'created': created,
        'patient_id': patient.id,
        'redirect_url': f'/doctor/patient/{patient.id}/' if doctor else '/doc_port/',
    })


# ============================================================================
# D. DETAILED PATIENT TELEMETRY VIEW & CUSTOM EXERCISE AUTO-INFERENCE
# ============================================================================

def patient_detail(request, patient_id):
    """
    Detailed Patient Telemetry View strictly scoped to doctor_id:
    1. AI Clinical Recovery Summary card & Extracted Medical Context
    2. SLS exercise workout history & interactive Chart.js telemetry
    3. Recent chatbot interaction logs & EmergencyAlerts
    4. Custom Exercise Video Upload / Webcam Recorder + MediaPipe & Groq Auto-Inference
    """
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return redirect('doctor_login')

    doctor = get_object_or_404(Doctor, id=doctor_id)
    patient = get_object_or_404(Patient, id=patient_id, doctor_id=doctor_id)

    telemetry = compute_patient_telemetry(patient, doctor_id=doctor_id)
    workouts = telemetry["workouts"]

    # Build chronological chart data (last 12 sessions)
    chronological_workouts = list(reversed(workouts[:12]))
    chart_data = []
    for w in chronological_workouts:
        good_only = max(0, w.good_reps - w.excellent_reps)
        chart_data.append({
            "date": w.date.strftime("%b %d"),
            "exercise_type": w.exercise_type,
            "total_reps": w.total_reps,
            "target_reps": w.target_reps,
            "excellent_reps": w.excellent_reps,
            "good_reps": good_only,
            "partial_reps": w.partial_reps,
            "quality_score": w.quality_score,
            "completion_percentage": min(100, w.completion_percentage),
            "duration_seconds": w.duration_seconds,
        })

    # Chatbot & system logs
    log_obj = Logs.objects.filter(patient=patient).first()
    chat_entries = parse_chat_logs(log_obj.text if log_obj else "")

    # Emergency alerts for this patient
    patient_alerts = EmergencyAlert.objects.filter(
        patient=patient,
        doctor_id=doctor_id
    ).order_by('-created_at')

    # Custom exercises prescribed for this patient
    custom_exercises = CustomExercise.objects.filter(
        patient=patient,
        doctor_id=doctor_id
    ).order_by('-created_at')

    context = {
        'doctor': doctor,
        'patient': patient,
        'telemetry': telemetry,
        'workouts': workouts[:25],
        'chart_data_json': json.dumps(chart_data),
        'chat_entries': chat_entries,
        'raw_log_text': log_obj.text if log_obj else '',
        'patient_alerts': patient_alerts,
        'custom_exercises': custom_exercises,
        'groq_model_name': os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    }
    return render(request, 'patient_detail.html', context)


@require_POST
def analyze_custom_exercise_video(request, patient_id):
    """
    Step 1 & 2 of Custom Exercise Pipeline:
    Accepts an uploaded or recorded MP4/WebM demo video, processes it with OpenCV + MediaPipe Pose
    to compute joint ROM across shoulders, elbows, hips, and knees, and queries Groq (qwen/qwen3.8-27b)
    to auto-infer exercise name, clinical description, target reps, sets, and difficulty.
    """
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return JsonResponse({'success': False, 'error': 'Doctor authentication required.'}, status=401)

    patient = get_object_or_404(Patient, id=patient_id, doctor_id=doctor_id)
    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'success': False, 'error': 'Please select or record an exercise video first.'}, status=400)

    suffix = os.path.splitext(video_file.name)[1] or '.mp4'
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        kinematics_data = analyze_exercise_video_kinematics(tmp_path)
        ai_suggestion = infer_exercise_with_groq(
            kinematics_data,
            patient=patient,
            filename=video_file.name
        )

        return JsonResponse({
            'success': True,
            'name': ai_suggestion['name'],
            'description': ai_suggestion['description'],
            'target_reps': ai_suggestion['target_reps'],
            'target_sets': ai_suggestion['target_sets'],
            'difficulty': ai_suggestion['difficulty'],
            'kinematics_summary': kinematics_data['kinematics_summary'],
            'joint_metrics': kinematics_data['joint_metrics'],
            'frames_analyzed': kinematics_data['frames_analyzed'],
            'primary_joint': kinematics_data.get('primary_joint', 'Shoulders'),
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Video analysis error: {str(e)}'}, status=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@require_POST
def prescribe_custom_exercise(request, patient_id):
    """
    Step 3 of Custom Exercise Pipeline:
    Saves the doctor-reviewed (or auto-inferred) CustomExercise for the patient.
    If a video is uploaded without prior kinematics_summary, automatically runs the
    MediaPipe Pose + Groq inference pipeline before saving.
    """
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return JsonResponse({'success': False, 'error': 'Doctor authentication required.'}, status=401)

    doctor = get_object_or_404(Doctor, id=doctor_id)
    patient = get_object_or_404(Patient, id=patient_id, doctor_id=doctor_id)

    video_file = request.FILES.get('video')
    name = request.POST.get('name', '').strip()
    description = request.POST.get('description', '').strip()
    kinematics_summary = request.POST.get('kinematics_summary', '').strip()
    difficulty = request.POST.get('difficulty', 'Medium').strip() or 'Medium'

    try:
        target_reps = max(1, int(request.POST.get('target_reps', 10)))
    except ValueError:
        target_reps = 10

    try:
        target_sets = max(1, int(request.POST.get('target_sets', 3)))
    except ValueError:
        target_sets = 3

    # If video was uploaded and doctor didn't pre-analyze (or left fields empty), auto-infer now
    if video_file and (not kinematics_summary or not name):
        suffix = os.path.splitext(video_file.name)[1] or '.mp4'
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in video_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            video_file.seek(0)
            kinematics_data = analyze_exercise_video_kinematics(tmp_path)
            if not kinematics_summary:
                kinematics_summary = kinematics_data['kinematics_summary']
            if not name or not description:
                inferred = infer_exercise_with_groq(kinematics_data, patient=patient, filename=video_file.name)
                if not name:
                    name = inferred['name']
                if not description:
                    description = inferred['description']
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if not name:
        return JsonResponse({'success': False, 'error': 'Exercise name is required.'}, status=400)

    custom_ex = CustomExercise.objects.create(
        doctor=doctor,
        patient=patient,
        name=name,
        description=description,
        target_reps=target_reps,
        target_sets=target_sets,
        difficulty=difficulty,
        video=video_file,
        kinematics_summary=kinematics_summary,
    )

    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', ''):
        return JsonResponse({
            'success': True,
            'exercise_id': custom_ex.id,
            'name': custom_ex.name,
            'video_url': custom_ex.video.url if custom_ex.video else '',
        })

    messages.success(request, f'Custom exercise "{custom_ex.name}" prescribed for {patient.name}!')
    return redirect('patient_detail', patient_id=patient.id)


@require_POST
def delete_custom_exercise(request, exercise_id):
    """Delete a custom exercise prescribed by the logged-in doctor"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return redirect('doctor_login')

    exercise = get_object_or_404(CustomExercise, id=exercise_id, doctor_id=doctor_id)
    patient_id = exercise.patient_id
    ex_name = exercise.name
    exercise.delete()
    messages.success(request, f'Removed custom exercise "{ex_name}".')
    return redirect('patient_detail', patient_id=patient_id)


@require_POST
def doctor_generate_summary(request, patient_id):
    """Generate or refresh AI Clinical Recovery Summary for a patient from the Doctor Portal"""
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:
        return JsonResponse({'success': False, 'error': 'Doctor authentication required.'}, status=401)

    patient = get_object_or_404(Patient, id=patient_id, doctor_id=doctor_id)
    telemetry = compute_patient_telemetry(patient, doctor_id=doctor_id)
    log_obj = Logs.objects.filter(patient=patient).first()
    log_text = log_obj.text[-6000:] if (log_obj and log_obj.text) else "No chatbot interactions recorded yet."

    workout_lines = []
    for w in telemetry["workouts"][:10]:
        workout_lines.append(
            f"- {w.date}: {w.exercise_type} | {w.total_reps}/{w.target_reps} reps "
            f"(Excellent: {w.excellent_reps}, Quality: {w.quality_score}/100, Duration: {w.duration_seconds}s)"
        )
    workout_summary_str = "\n".join(workout_lines) if workout_lines else "No exercise sessions recorded yet."

    llm = get_groq_llm(temperature=0.25, max_tokens=900)
    if llm:
        try:
            system_prompt = """You are a senior clinical rehabilitation AI assistant for ReViveCare.
Create a concise, structured clinical progress summary for the attending doctor.
Include these sections using clean bullet points:
1. Clinical Condition & Baseline Context
2. Rehabilitation & Kinematic Adherence (compliance, quality scores, ROM progression)
3. Reported Symptoms & Chatbot Triage Insights
4. Clinical Recommendations & Next Steps"""
            human_prompt = f"""PATIENT NAME: {patient.name}
CLINICAL CONTEXT / EXTRACTED REPORT:
{patient.info}

WORKOUT TELEMETRY (Streak: {telemetry['streak']} days, Compliance: {telemetry['compliance_rate']}%, Avg Form Score: {telemetry['avg_quality']}/100):
{workout_summary_str}

RECENT PATIENT LOGS & CHATBOT INTERACTIONS:
{log_text}"""
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])
            summary = re.sub(r'<think>[\s\S]*?</think>', '', response.content).strip()
        except Exception as e:
            summary = (
                f"Clinical Summary for {patient.name}\n"
                f"• Total Sessions Completed: {telemetry['total_workouts']} (Compliance: {telemetry['compliance_rate']}%)\n"
                f"• Average Kinematic Quality Score: {telemetry['avg_quality']}/100 | Active Streak: {telemetry['streak']} days\n"
                f"• Baseline Context: {patient.info[:400]}"
            )
    else:
        summary = (
            f"Clinical Summary for {patient.name}\n"
            f"• Total Sessions Completed: {telemetry['total_workouts']} (Compliance: {telemetry['compliance_rate']}%)\n"
            f"• Average Kinematic Quality Score: {telemetry['avg_quality']}/100 | Active Streak: {telemetry['streak']} days\n"
            f"• Baseline Context: {patient.info[:400]}"
        )

    patient.summary = summary
    patient.updated_at = timezone.now()
    patient.save(update_fields=['summary', 'updated_at'])

    return JsonResponse({
        'success': True,
        'summary': summary,
        'updated_at': patient.updated_at.strftime('%b %d, %Y %H:%M'),
    })
