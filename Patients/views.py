# patient/views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Sum, Avg, Count, Max
from django.core.mail import send_mail
from django.conf import settings
from twilio.rest import Client
from django.utils import timezone
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date, timedelta
from .models import Patient, SLS, Logs
from Doctor.models import Doctor, CustomExercise, EmergencyAlert
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()  

# Import LangChain components
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not installed. Chatbot will use mock responses.")

# Configuration
# by default, it loads from a file named ".env" in the current directory

# Set the API key from the environment, if not already set
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
os.environ.setdefault("SMTP_SERVER", "smtp.gmail.com")
os.environ.setdefault("SMTP_PORT", "587")
os.environ.setdefault("EMAIL_SENDER", "animeguy055@gmail.com")
os.environ.setdefault("EMAIL_PASSWORD", "zuby ghtn xfbt itco")
os.environ.setdefault("DOCTOR_EMAIL", "sarthak.molu08@gmail.com")

SYSTEM_PROMPT = """You are ReViveCare, an AI-powered post-discharge recovery support chatbot.
Your role is to support patients emotionally and informationally during recovery after surgery or injury.
You are NOT a doctor and you MUST NOT diagnose, prescribe medication, or alter treatment plans.

You will be given full patient context provided by their doctor.
Use this context to answer user queries accurately, calmly, and empathetically.

Your goals:
1. Reassure and guide the patient using simple, non-alarming language
2. Compare their reported symptoms against their specific medical context
3. Determine if symptoms align with expected recovery for THEIR condition
4. Encourage appropriate next steps without diagnosing
5. Detect potentially serious situations based on THEIR specific medical context

IMPORTANT RULES:
- Never provide a diagnosis
- Never prescribe or change medications
- Never claim certainty about outcomes
- Never dismiss the user's concern
- Be calm, supportive, and empathetic
- Base all assessments on the patient's specific condition and context provided by their doctor

You must return EXACTLY a valid JSON object for every user message with these keys:
1. patient_response: A clear, supportive answer to the user's query based on their specific medical context
2. seriousness_score: A single floating-point number between 0.0 and 1.0

SERIOUSNESS SCORING GUIDELINES:
0.00 – 0.30 → Symptoms align with expected recovery for their specific condition, reassurance sufficient
0.31 – 0.60 → Monitor closely, routine follow-up advised based on their condition
0.61 – 0.75 → Concerning for their specific condition, recommend contacting doctor soon
> 0.75 → Potential red flag for their specific condition, immediate escalation required

When scoring seriousness:
- Consider the patient's specific condition and what is normal for THEIR recovery
- Compare symptoms against expected vs. concerning signs for THEIR medical context
- Account for their prescribed medications and potential side effects
- Consider their recovery timeline and stage

If seriousness_score > 0.75:
- In your patient_response, calmly advise the patient to contact their doctor or hospital immediately
- Do NOT use panic-inducing language unless absolutely necessary
- Reference their specific condition when explaining why they should seek help

STRICT OUTPUT FORMAT (valid JSON only, no extra text):
{{
  "patient_response": "your response here",
  "seriousness_score": 0.0
}}

PATIENT MEDICAL CONTEXT:
{patient_context}
"""

# ============================================================================
# PATIENT AUTHENTICATION & DASHBOARD VIEWS
# ============================================================================

def login(request):
    """Patient login - email only authentication"""
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        
        patient = Patient.objects.filter(email__iexact=email).first()
        if patient:
            # Email found - log them in
            request.session['patient_id'] = patient.id
            request.session['patient_email'] = patient.email
            request.session['patient_name'] = patient.name
            
            messages.success(request, f'Welcome back, {patient.name}!')
            return redirect('patient_dashboard')
        else:
            # Email not found - redirect to home
            messages.error(
                request, 
                'Doctor has not updated this email yet. Please come back later.'
            )
            return redirect('home')
    
    # GET request - show login form
    return render(request, 'login_page.html')


def patient_dashboard(request):
    """Patient dashboard"""
    patient_id = request.session.get('patient_id')
    if not patient_id:
        messages.error(request, 'Please log in to access the patient portal.')
        return redirect('login')
    
    try:
        patient = Patient.objects.get(id=patient_id)
        return render(request, 'patient_dashboard.html', {'patient': patient})
    except Patient.DoesNotExist:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('login')


def patient_logout(request):
    """Logout patient"""
    for key in ('patient_id', 'patient_email', 'patient_name', 'chat_history'):
        request.session.pop(key, None)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')

# ============================================================================
# CHATBOT VIEWS
# ============================================================================

def chatbot(request):
    """Chatbot page"""
    patient_id = request.session.get('patient_id')
    if not patient_id:
        messages.error(request, 'Please log in to access the chatbot.')
        return redirect('login')
    
    try:
        patient = Patient.objects.get(id=patient_id)
        
        # Initialize conversation history in session if not exists
        if 'chat_history' not in request.session:
            request.session['chat_history'] = []
        
        return render(request, 'chatbot.html', {'patient': patient})
    except Patient.DoesNotExist:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('login')


def initialize_model():
    """Initialize the ChatGroq model"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model_name,
        temperature=0.3,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    )


def parse_response(raw_response):
    """Parse AI response to extract patient_response and seriousness_score"""
    try:
        response_text = raw_response.content.strip()
        response_text = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            patient_response = parsed.get("patient_response", "")
            seriousness_score = float(parsed.get("seriousness_score", 0.0))
            
            # Clamp seriousness score between 0 and 1
            seriousness_score = max(0.0, min(1.0, seriousness_score))
            
            return patient_response, seriousness_score
        else:
            return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question.", 0.0
            
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Parsing error: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question.", 0.0


def send_email_alert(patient, user_query, patient_response, seriousness_score):
    """Send email alert to doctor when seriousness score is high"""
    try:
        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = int(os.environ.get("SMTP_PORT"))
        sender_email = os.environ.get("EMAIL_SENDER")
        sender_password = os.environ.get("EMAIL_PASSWORD")
        doctor_email = os.environ.get("DOCTOR_EMAIL")
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = doctor_email
        msg['Subject'] = f"🚨 URGENT - Patient Alert: {patient.name} (Score: {seriousness_score:.2f})"
        
        email_body = f"""
URGENT PATIENT ALERT - ReViveCare System
============================================

Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Patient Name: {patient.name}
Patient Email: {patient.email}
Seriousness Score: {seriousness_score:.2f}

LATEST PATIENT QUERY:
{user_query}

SYSTEM RESPONSE TO PATIENT:
{patient_response}

PATIENT MEDICAL CONTEXT:
{patient.info}

RECOMMENDATION:
Please contact the patient immediately to assess their condition and provide appropriate medical guidance.

This is an automated alert from the ReViveCare post-discharge monitoring system.
"""
        
        msg.attach(MIMEText(email_body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, doctor_email, text)
        server.quit()
        
        print(f"[OK] EMAIL ALERT SENT TO DOCTOR: {doctor_email}")
        TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
        TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # Your Twilio phone number
        TO_NUMBER = os.getenv("TO_NUMBER")
            
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
        call = client.calls.create(
            to=TO_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            twiml=f"<Response><Say voice='alice'>This is an emergency please call {patient.name}</Say></Response>"
        )
            
        context = {
            'call_sid': call.sid,
            'status': 'Call initiated successfully',
            'phone_number': TO_NUMBER,
            'message': f"IVR call is being made to say  'This is an emergency please call {patient.name}'"
        }
        
    except Exception as e:
        print(f"[WARN] EMAIL SEND FAILED: {str(e)}")
        context = {
            'status': 'Error making call',
            'error': str(e)
        }

@require_http_methods(["POST"])
def chatbot_send(request):
    """Handle chatbot message sending"""
    patient_id = request.session.get('patient_id')
    if not patient_id:
        return JsonResponse({'success': False, 'error': 'Not authenticated'})
    
    try:
        # Get patient
        patient = Patient.objects.get(id=patient_id)
        
        # Parse request
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({'success': False, 'error': 'Empty message'})
        
        # Get or initialize chat history
        chat_history = request.session.get('chat_history', [])
        
        # Prepare system prompt with patient context
        system_prompt = SYSTEM_PROMPT.format(patient_context=patient.info)
        
        if LANGCHAIN_AVAILABLE:
            # Use real AI model
            llm = initialize_model()
            
            # Build messages
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history
            for msg in chat_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                else:
                    messages.append(AIMessage(content=msg['content']))
            
            # Add current query
            messages.append(HumanMessage(content=user_message))
            
            # Get AI response
            response = llm.invoke(messages)
            patient_response, seriousness_score = parse_response(response)
            
        else:
            # Mock response for development
            patient_response = f"I understand your concern about: '{user_message}'. Based on your recovery plan, this seems normal. However, I'm currently in development mode. Please consult your doctor for specific medical advice."
            seriousness_score = 0.2
        
        # Update chat history
        chat_history.append({'role': 'user', 'content': user_message})
        chat_history.append({'role': 'assistant', 'content': patient_response})

        append_to_patient_log(patient, f" Patient : {user_message}")
        append_to_patient_log(patient, f" AI  : {patient_response}")
        append_to_patient_log(patient, '\n')
        
        # Keep only last 20 messages to prevent session bloat
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        request.session['chat_history'] = chat_history
        request.session.modified = True
        
        # Record EmergencyAlert in DB and send email alert if high seriousness (> 0.75)
        if seriousness_score > 0.75:
            assigned_doctor = patient.doctor or Doctor.objects.first()
            EmergencyAlert.objects.create(
                doctor=assigned_doctor,
                patient=patient,
                user_query=user_message,
                ai_response=patient_response,
                seriousness_score=seriousness_score,
                is_reviewed=False,
            )
            send_email_alert(patient, user_message, patient_response, seriousness_score)
        
        return JsonResponse({
            'success': True,
            'response': patient_response,
            'seriousness_score': seriousness_score
        })
        
    except Patient.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Patient not found'})
    except Exception as e:
        print(f"Error in chatbot_send: {e}")
        return JsonResponse({'success': False, 'error': 'Server error'})


# ============================================================================
# SHARED KINEMATICS & POSE RENDERING UTILITIES
# ============================================================================

def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c where b is the vertex)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def get_landmark_coords(landmarks, landmark_id):
    """Extract x, y coordinates from a landmark"""
    landmark = landmarks[landmark_id]
    return [landmark.x, landmark.y]


def calculate_shoulder_elevation(current_shoulder, initial_shoulder):
    """Calculate vertical shoulder movement (shrugging detection)"""
    return abs(current_shoulder[1] - initial_shoulder[1])


def calculate_shoulder_drift(current_shoulder, initial_shoulder):
    """Calculate Euclidean 2D shoulder drift from initial position (for Bicep Curls)"""
    return float(np.sqrt(
        (current_shoulder[0] - initial_shoulder[0]) ** 2 +
        (current_shoulder[1] - initial_shoulder[1]) ** 2
    ))


def draw_studio_skeleton(image, results, joint_badges=None, has_warning=False):
    """
    Render a clean, hospital-grade studio skeletal overlay on the webcam stream.
    Draws only anti-aliased glowing skeletal lines, joint nodes, and subtle joint angle tags.
    All telemetry cards and status banners are rendered via HTML/CSS glassmorphism overlays.
    """
    if not results.pose_landmarks:
        return

    h, w, _ = image.shape
    landmarks = results.pose_landmarks.landmark
    mp_pose = mp.solutions.pose

    # Brand palette in BGR:
    # Medical Teal (#00A8A8) -> BGR (168, 168, 0)
    # Light Teal Glow (#80CBC4) -> BGR (196, 203, 128)
    # Deep Navy (#1E3A5F) -> BGR (95, 58, 30)
    # Warning Amber (#F59E0B) -> BGR (11, 158, 245)
    primary_line_color = (11, 158, 245) if has_warning else (168, 168, 0)
    glow_line_color = (80, 190, 255) if has_warning else (210, 225, 120)
    node_fill_color = (95, 58, 30)
    node_ring_color = (255, 255, 255) if not has_warning else (80, 200, 255)

    # Create overlay for soft alpha glow
    overlay = image.copy()

    # Draw torso & limb connections with anti-aliased lines
    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
        # Skip facial connections (0-10) to keep the face clean and clinical
        if start_idx <= 10 and end_idx <= 10:
            continue
        lm_start = landmarks[start_idx]
        lm_end = landmarks[end_idx]
        if lm_start.visibility < 0.35 or lm_end.visibility < 0.35:
            continue

        pt1 = (int(lm_start.x * w), int(lm_start.y * h))
        pt2 = (int(lm_end.x * w), int(lm_end.y * h))

        # Outer glow stroke on overlay
        cv2.line(overlay, pt1, pt2, glow_line_color, 8, cv2.LINE_AA)
        # Core crisp stroke on main image
        cv2.line(image, pt1, pt2, primary_line_color, 3, cv2.LINE_AA)

    # Blend glow overlay
    cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)

    # Draw major anatomical joint nodes (skip face nodes 1-10)
    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for idx in key_joints:
        lm = landmarks[idx]
        if lm.visibility < 0.35:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 8, glow_line_color, -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 6, node_fill_color, -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 6, node_ring_color, 2, cv2.LINE_AA)

    # Draw subtle, anti-aliased joint angle tags next to active joints
    if joint_badges:
        for coords, label in joint_badges:
            jx, jy = int(coords[0] * w), int(coords[1] * h)
            # Offset slightly from the joint node
            tx, ty = jx + 14, jy - 8
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            # Small pill background
            pad_x, pad_y = 7, 5
            x1 = max(0, tx - pad_x)
            y1 = max(0, ty - th - pad_y)
            x2 = min(w, tx + tw + pad_x)
            y2 = min(h, ty + pad_y)
            sub = image[y1:y2, x1:x2]
            if sub.size > 0:
                pill_bg = np.full(sub.shape, (45, 30, 15), dtype=np.uint8)
                cv2.addWeighted(pill_bg, 0.75, sub, 0.25, 0, sub)
                image[y1:y2, x1:x2] = sub
                cv2.rectangle(image, (x1, y1), (x2, y2), primary_line_color, 1, cv2.LINE_AA)
            cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================
# 1. SIDE LATERAL RAISE (SR) - METRICS, STATE & STREAMING
# ============================================================================

LEFT_ARM_METRICS = {
    "down_threshold": 35,
    "up_threshold": 100,
    "excellent_peak": 105,
    "good_peak": 95,
    "min_rom": 60,
    "target_elbow": 176,
    "max_elbow_variance": 5,
    "max_shoulder_elevation": 0.025,
}

RIGHT_ARM_METRICS = {
    "down_threshold": 35,
    "up_threshold": 100,
    "excellent_peak": 105,
    "good_peak": 95,
    "min_rom": 60,
    "target_elbow": 175,
    "max_elbow_variance": 5,
    "max_shoulder_elevation": 0.025,
}

SYMMETRY_METRICS = {
    "excellent_threshold": 5,
    "good_threshold": 10,
    "max_allowed": 15,
}

def _default_sr_rep():
    return {
        "left_min_angle": 180,
        "left_max_angle": 0,
        "left_elbow_angles": [],
        "left_shoulder_elevation": 0,
        "right_min_angle": 180,
        "right_max_angle": 0,
        "right_elbow_angles": [],
        "right_shoulder_elevation": 0,
        "max_asymmetry": 0,
    }

workout_state = {
    'exercise_type': 'Side Lateral Raise',
    'active': False,
    'complete': False,
    'counter': 0,
    'stage': None,
    'excellent_reps': 0,
    'good_reps': 0,
    'partial_reps': 0,
    'target_reps': 12,
    'current_rep': _default_sr_rep(),
    'initial_left_shoulder': None,
    'initial_right_shoulder': None,
    'form_warning': '',
    'left_shoulder_angle': 0,
    'right_shoulder_angle': 0,
    'left_elbow_angle': 0,
    'right_elbow_angle': 0,
    'asymmetry': 0,
    'shrug_level': 0,
    'symmetry_score': 100,
    'shrug_score': 100,
    'start_time': None,
    'workout_record': None,
}


def generate_frames():
    """Generate clean studio video frames with pose detection and rep counting for Side Lateral Raise"""
    global workout_state
    
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for natural mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            joint_badges = []
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Extract key landmarks
                    left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                    left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
                    left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
                    
                    right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    right_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                    right_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
                    right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
                    
                    # Calculate live angles even before start so the patient sees live bio-telemetry
                    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                    right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    workout_state['left_shoulder_angle'] = left_shoulder_angle
                    workout_state['right_shoulder_angle'] = right_shoulder_angle
                    workout_state['left_elbow_angle'] = left_elbow_angle
                    workout_state['right_elbow_angle'] = right_elbow_angle
                    
                    asymmetry = abs(left_shoulder_angle - right_shoulder_angle)
                    workout_state['asymmetry'] = asymmetry
                    workout_state['symmetry_score'] = max(0, min(100, int(100 - (asymmetry / 20.0) * 100)))
                    
                    joint_badges = [
                        (left_shoulder, f"{int(left_shoulder_angle)} deg"),
                        (right_shoulder, f"{int(right_shoulder_angle)} deg"),
                    ]
                    
                    # Active workout state machine
                    if workout_state['active'] and not workout_state['complete']:
                        if workout_state['initial_left_shoulder'] is None:
                            workout_state['initial_left_shoulder'] = left_shoulder
                            workout_state['initial_right_shoulder'] = right_shoulder
                        
                        left_shoulder_elevation = calculate_shoulder_elevation(
                            left_shoulder, workout_state['initial_left_shoulder'])
                        right_shoulder_elevation = calculate_shoulder_elevation(
                            right_shoulder, workout_state['initial_right_shoulder'])
                        
                        max_elev = max(left_shoulder_elevation, right_shoulder_elevation)
                        workout_state['shrug_level'] = round(max_elev * 1000, 1)
                        workout_state['shrug_score'] = max(0, min(100, int(100 - (max_elev / 0.04) * 100)))
                        
                        current_rep = workout_state['current_rep']
                        avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
                        
                        current_rep['left_min_angle'] = min(current_rep['left_min_angle'], left_shoulder_angle)
                        current_rep['left_max_angle'] = max(current_rep['left_max_angle'], left_shoulder_angle)
                        current_rep['right_min_angle'] = min(current_rep['right_min_angle'], right_shoulder_angle)
                        current_rep['right_max_angle'] = max(current_rep['right_max_angle'], right_shoulder_angle)
                        current_rep['max_asymmetry'] = max(current_rep['max_asymmetry'], asymmetry)
                        
                        if workout_state['stage'] == "up":
                            current_rep['left_elbow_angles'].append(left_elbow_angle)
                            current_rep['right_elbow_angles'].append(right_elbow_angle)
                            current_rep['left_shoulder_elevation'] = max(
                                current_rep['left_shoulder_elevation'], left_shoulder_elevation)
                            current_rep['right_shoulder_elevation'] = max(
                                current_rep['right_shoulder_elevation'], right_shoulder_elevation)
                        
                        # Rep detection
                        if avg_shoulder_angle <= LEFT_ARM_METRICS["down_threshold"] and workout_state['stage'] == "up":
                            left_rom = current_rep['left_max_angle'] - current_rep['left_min_angle']
                            right_rom = current_rep['right_max_angle'] - current_rep['right_min_angle']
                            avg_rom = (left_rom + right_rom) / 2
                            avg_peak = (current_rep['left_max_angle'] + current_rep['right_max_angle']) / 2
                            
                            left_elbow_std = np.std(current_rep['left_elbow_angles']) if current_rep['left_elbow_angles'] else 0
                            right_elbow_std = np.std(current_rep['right_elbow_angles']) if current_rep['right_elbow_angles'] else 0
                            elbow_consistent = (left_elbow_std < LEFT_ARM_METRICS["max_elbow_variance"] and 
                                              right_elbow_std < RIGHT_ARM_METRICS["max_elbow_variance"])
                            
                            shoulder_stable = (current_rep['left_shoulder_elevation'] <= LEFT_ARM_METRICS["max_shoulder_elevation"] and
                                             current_rep['right_shoulder_elevation'] <= RIGHT_ARM_METRICS["max_shoulder_elevation"])
                            
                            symmetry_excellent = current_rep['max_asymmetry'] < SYMMETRY_METRICS["excellent_threshold"]
                            symmetry_good = current_rep['max_asymmetry'] < SYMMETRY_METRICS["good_threshold"]
                            
                            if (avg_peak >= LEFT_ARM_METRICS["excellent_peak"] and 
                                avg_rom >= LEFT_ARM_METRICS["min_rom"] and 
                                elbow_consistent and shoulder_stable and symmetry_excellent):
                                workout_state['excellent_reps'] += 1
                                workout_state['good_reps'] += 1
                            elif (avg_peak >= LEFT_ARM_METRICS["good_peak"] and 
                                  avg_rom >= LEFT_ARM_METRICS["min_rom"] and 
                                  symmetry_good):
                                workout_state['good_reps'] += 1
                            else:
                                workout_state['partial_reps'] += 1
                            
                            workout_state['stage'] = "down"
                            workout_state['counter'] += 1
                            workout_state['current_rep'] = _default_sr_rep()
                            
                            if workout_state['counter'] >= workout_state['target_reps']:
                                workout_state['complete'] = True
                                workout_state['active'] = False
                        
                        elif avg_shoulder_angle >= RIGHT_ARM_METRICS["up_threshold"] and workout_state['stage'] != "up":
                            workout_state['stage'] = "up"
                        elif workout_state['stage'] is None and avg_shoulder_angle <= LEFT_ARM_METRICS["down_threshold"]:
                            workout_state['stage'] = "down"
                        
                        # Form warnings
                        form_warning = ""
                        if asymmetry > SYMMETRY_METRICS["max_allowed"]:
                            form_warning = "⚠ Keep arms even (bilateral asymmetry detected)"
                        elif (left_shoulder_elevation > LEFT_ARM_METRICS["max_shoulder_elevation"] or 
                              right_shoulder_elevation > RIGHT_ARM_METRICS["max_shoulder_elevation"]):
                            form_warning = "⚠ Don't shrug shoulders (relax upper traps)"
                        elif workout_state['stage'] == "up":
                            avg_elbow = (left_elbow_angle + right_elbow_angle) / 2
                            if avg_elbow < 160:
                                form_warning = "⚠ Keep elbows straighter during lift"
                        
                        workout_state['form_warning'] = form_warning
                except Exception:
                    pass
            
            # Draw clean studio skeleton without any solid gray boxes or pixelated HUD text
            draw_studio_skeleton(
                image,
                results,
                joint_badges=joint_badges,
                has_warning=bool(workout_state.get('form_warning'))
            )
            
            ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


# ============================================================================
# 2. BICEP CURLS (BC) - METRICS, STATE & STREAMING (PORTED FROM bc6.py)
# ============================================================================

BC_LEFT_METRICS = {
    "down_threshold": 140,
    "up_threshold": 70,
    "excellent_peak": 45,
    "good_peak": 70,
    "min_rom": 60,
    "max_shoulder_drift": 0.050,
}

BC_RIGHT_METRICS = {
    "down_threshold": 140,
    "up_threshold": 70,
    "excellent_peak": 45,
    "good_peak": 70,
    "min_rom": 60,
    "max_shoulder_drift": 0.050,
}

def _default_bc_arm_rep():
    return {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0.0}

bc_state = {
    'exercise_type': 'Bicep Curls',
    'active': False,
    'complete': False,
    'counter': 0,
    'target_reps': 10,
    'stage': None,
    'current_arm': 'LEFT',   # 'LEFT', 'RIGHT', or 'BOTH'
    'arm_mode': 'SEQUENTIAL', # 'SEQUENTIAL' (L then R as in bc6.py) or 'BILATERAL'
    'left_counter': 0,
    'right_counter': 0,
    'left_stage': None,
    'right_stage': None,
    'left_initial_shoulder': None,
    'right_initial_shoulder': None,
    'left_current_rep': _default_bc_arm_rep(),
    'right_current_rep': _default_bc_arm_rep(),
    'excellent_reps': 0,
    'good_reps': 0,
    'partial_reps': 0,
    'left_elbow_angle': 0,
    'right_elbow_angle': 0,
    'left_drift': 0.0,
    'right_drift': 0.0,
    'stability_score': 100,
    'rom_score': 0,
    'form_warning': '',
    'start_time': None,
    'workout_record': None,
}


def _evaluate_bc_rep(rep_data, metrics):
    """Evaluate Bicep Curl rep quality using exact thresholds from execrise/bc6.py"""
    rom = max(0.0, rep_data["max_angle"] - rep_data["min_angle"])
    peak_angle = rep_data["min_angle"]
    shoulder_stable = rep_data["shoulder_drift"] < metrics["max_shoulder_drift"]

    if rom >= metrics["min_rom"] and shoulder_stable:
        if peak_angle <= metrics["excellent_peak"]:
            return "EXCELLENT", rom
        return "GOOD", rom
    elif rom >= (metrics["min_rom"] * 0.75):
        return "GOOD", rom
    else:
        return "PARTIAL", rom


def generate_bc_frames():
    """Generate clean studio video frames for Bicep Curls with separate L/R arm & shoulder drift tracking"""
    global bc_state
    
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            joint_badges = []
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    l_sh = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    l_el = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                    l_wr = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
                    
                    r_sh = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    r_el = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                    r_wr = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
                    
                    l_angle = calculate_angle(l_sh, l_el, l_wr)
                    r_angle = calculate_angle(r_sh, r_el, r_wr)
                    
                    bc_state['left_elbow_angle'] = l_angle
                    bc_state['right_elbow_angle'] = r_angle
                    
                    joint_badges = [
                        (l_el, f"L {int(l_angle)} deg"),
                        (r_el, f"R {int(r_angle)} deg"),
                    ]
                    
                    if bc_state['active'] and not bc_state['complete']:
                        if bc_state['left_initial_shoulder'] is None:
                            bc_state['left_initial_shoulder'] = l_sh.copy()
                        if bc_state['right_initial_shoulder'] is None:
                            bc_state['right_initial_shoulder'] = r_sh.copy()
                        
                        l_drift = calculate_shoulder_drift(l_sh, bc_state['left_initial_shoulder'])
                        r_drift = calculate_shoulder_drift(r_sh, bc_state['right_initial_shoulder'])
                        bc_state['left_drift'] = round(l_drift, 3)
                        bc_state['right_drift'] = round(r_drift, 3)
                        
                        max_drift = max(l_drift, r_drift)
                        bc_state['stability_score'] = max(0, min(100, int(100 - (max_drift / 0.08) * 100)))
                        
                        form_warning = ""
                        current_arm = bc_state['current_arm']
                        
                        # Track LEFT arm if current_arm is LEFT or BOTH
                        if current_arm in ('LEFT', 'BOTH'):
                            l_rep = bc_state['left_current_rep']
                            l_rep['min_angle'] = min(l_rep['min_angle'], l_angle)
                            l_rep['max_angle'] = max(l_rep['max_angle'], l_angle)
                            l_rep['shoulder_drift'] = max(l_rep['shoulder_drift'], l_drift)
                            
                            if l_drift > BC_LEFT_METRICS['max_shoulder_drift']:
                                form_warning = "⚠ Keep LEFT shoulder stable (avoid swinging)"
                            
                            if l_angle > BC_LEFT_METRICS['down_threshold']:
                                bc_state['left_stage'] = 'down'
                            elif l_angle < BC_LEFT_METRICS['up_threshold'] and bc_state['left_stage'] == 'down':
                                quality, rom = _evaluate_bc_rep(l_rep, BC_LEFT_METRICS)
                                bc_state['rom_score'] = min(100, int((rom / 110.0) * 100))
                                if quality == "EXCELLENT":
                                    bc_state['excellent_reps'] += 1
                                    bc_state['good_reps'] += 1
                                elif quality == "GOOD":
                                    bc_state['good_reps'] += 1
                                else:
                                    bc_state['partial_reps'] += 1
                                
                                bc_state['left_stage'] = 'up'
                                bc_state['left_counter'] += 1
                                bc_state['counter'] = bc_state['left_counter'] + bc_state['right_counter']
                                bc_state['left_current_rep'] = _default_bc_arm_rep()
                                
                                # Auto-advance to RIGHT arm in SEQUENTIAL mode when left target reached
                                per_arm_target = max(1, bc_state['target_reps'] // 2) if bc_state['arm_mode'] == 'SEQUENTIAL' else bc_state['target_reps']
                                if bc_state['arm_mode'] == 'SEQUENTIAL' and bc_state['left_counter'] >= per_arm_target:
                                    bc_state['current_arm'] = 'RIGHT'
                        
                        # Track RIGHT arm if current_arm is RIGHT or BOTH
                        if bc_state['current_arm'] in ('RIGHT', 'BOTH'):
                            r_rep = bc_state['right_current_rep']
                            r_rep['min_angle'] = min(r_rep['min_angle'], r_angle)
                            r_rep['max_angle'] = max(r_rep['max_angle'], r_angle)
                            r_rep['shoulder_drift'] = max(r_rep['shoulder_drift'], r_drift)
                            
                            if r_drift > BC_RIGHT_METRICS['max_shoulder_drift']:
                                form_warning = "⚠ Keep RIGHT shoulder stable (avoid swinging)"
                            
                            if r_angle > BC_RIGHT_METRICS['down_threshold']:
                                bc_state['right_stage'] = 'down'
                            elif r_angle < BC_RIGHT_METRICS['up_threshold'] and bc_state['right_stage'] == 'down':
                                quality, rom = _evaluate_bc_rep(r_rep, BC_RIGHT_METRICS)
                                bc_state['rom_score'] = min(100, int((rom / 110.0) * 100))
                                if quality == "EXCELLENT":
                                    bc_state['excellent_reps'] += 1
                                    bc_state['good_reps'] += 1
                                elif quality == "GOOD":
                                    bc_state['good_reps'] += 1
                                else:
                                    bc_state['partial_reps'] += 1
                                
                                bc_state['right_stage'] = 'up'
                                bc_state['right_counter'] += 1
                                bc_state['counter'] = bc_state['left_counter'] + bc_state['right_counter']
                                bc_state['right_current_rep'] = _default_bc_arm_rep()
                        
                        # Update unified stage & completion
                        active_stage = bc_state['left_stage'] if bc_state['current_arm'] == 'LEFT' else bc_state['right_stage']
                        bc_state['stage'] = active_stage
                        bc_state['form_warning'] = form_warning
                        
                        if bc_state['counter'] >= bc_state['target_reps']:
                            bc_state['complete'] = True
                            bc_state['active'] = False
                except Exception:
                    pass
            
            draw_studio_skeleton(
                image,
                results,
                joint_badges=joint_badges,
                has_warning=bool(bc_state.get('form_warning'))
            )
            
            ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()


# ============================================================================
# 3. ARM RAISES (AR) - METRICS, STATE & STREAMING
# ============================================================================

AR_METRICS = {
    "down_threshold": 35,
    "up_threshold": 140,
    "excellent_peak": 155,
    "good_peak": 140,
    "min_rom": 95,
    "max_shoulder_elevation": 0.030,
    "max_asymmetry": 15,
}

def _default_ar_rep():
    return {
        "left_min_angle": 180,
        "left_max_angle": 0,
        "right_min_angle": 180,
        "right_max_angle": 0,
        "max_elevation": 0.0,
        "max_asymmetry": 0.0,
    }

ar_state = {
    'exercise_type': 'Arm Raises',
    'active': False,
    'complete': False,
    'counter': 0,
    'target_reps': 10,
    'stage': None,
    'excellent_reps': 0,
    'good_reps': 0,
    'partial_reps': 0,
    'current_rep': _default_ar_rep(),
    'initial_left_shoulder': None,
    'initial_right_shoulder': None,
    'left_shoulder_angle': 0,
    'right_shoulder_angle': 0,
    'asymmetry': 0,
    'symmetry_score': 100,
    'shrug_score': 100,
    'form_warning': '',
    'start_time': None,
    'workout_record': None,
}


def generate_ar_frames():
    """Generate clean studio video frames for Overhead Arm Raises with shrug & symmetry tracking"""
    global ar_state
    
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            joint_badges = []
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    l_sh = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    l_el = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                    l_hp = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
                    
                    r_sh = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    r_el = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                    r_hp = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
                    
                    l_angle = calculate_angle(l_hp, l_sh, l_el)
                    r_angle = calculate_angle(r_hp, r_sh, r_el)
                    
                    ar_state['left_shoulder_angle'] = l_angle
                    ar_state['right_shoulder_angle'] = r_angle
                    
                    asym = abs(l_angle - r_angle)
                    ar_state['asymmetry'] = asym
                    ar_state['symmetry_score'] = max(0, min(100, int(100 - (asym / 25.0) * 100)))
                    
                    joint_badges = [
                        (l_sh, f"L {int(l_angle)} deg"),
                        (r_sh, f"R {int(r_angle)} deg"),
                    ]
                    
                    if ar_state['active'] and not ar_state['complete']:
                        if ar_state['initial_left_shoulder'] is None:
                            ar_state['initial_left_shoulder'] = l_sh.copy()
                            ar_state['initial_right_shoulder'] = r_sh.copy()
                        
                        l_elev = calculate_shoulder_elevation(l_sh, ar_state['initial_left_shoulder'])
                        r_elev = calculate_shoulder_elevation(r_sh, ar_state['initial_right_shoulder'])
                        max_elev = max(l_elev, r_elev)
                        ar_state['shrug_score'] = max(0, min(100, int(100 - (max_elev / 0.045) * 100)))
                        
                        avg_angle = (l_angle + r_angle) / 2.0
                        cur = ar_state['current_rep']
                        cur['left_min_angle'] = min(cur['left_min_angle'], l_angle)
                        cur['left_max_angle'] = max(cur['left_max_angle'], l_angle)
                        cur['right_min_angle'] = min(cur['right_min_angle'], r_angle)
                        cur['right_max_angle'] = max(cur['right_max_angle'], r_angle)
                        cur['max_elevation'] = max(cur['max_elevation'], max_elev)
                        cur['max_asymmetry'] = max(cur['max_asymmetry'], asym)
                        
                        if avg_angle >= AR_METRICS['up_threshold'] and ar_state['stage'] != 'up':
                            ar_state['stage'] = 'up'
                        elif avg_angle <= AR_METRICS['down_threshold'] and ar_state['stage'] == 'up':
                            avg_peak = (cur['left_max_angle'] + cur['right_max_angle']) / 2.0
                            avg_rom = ((cur['left_max_angle'] - cur['left_min_angle']) +
                                       (cur['right_max_angle'] - cur['right_min_angle'])) / 2.0
                            shrug_ok = cur['max_elevation'] <= AR_METRICS['max_shoulder_elevation']
                            sym_ok = cur['max_asymmetry'] <= AR_METRICS['max_asymmetry']
                            
                            if avg_peak >= AR_METRICS['excellent_peak'] and avg_rom >= AR_METRICS['min_rom'] and shrug_ok and sym_ok:
                                ar_state['excellent_reps'] += 1
                                ar_state['good_reps'] += 1
                            elif avg_peak >= AR_METRICS['good_peak'] and avg_rom >= (AR_METRICS['min_rom'] * 0.8):
                                ar_state['good_reps'] += 1
                            else:
                                ar_state['partial_reps'] += 1
                            
                            ar_state['stage'] = 'down'
                            ar_state['counter'] += 1
                            ar_state['current_rep'] = _default_ar_rep()
                            
                            if ar_state['counter'] >= ar_state['target_reps']:
                                ar_state['complete'] = True
                                ar_state['active'] = False
                        elif ar_state['stage'] is None and avg_angle <= AR_METRICS['down_threshold']:
                            ar_state['stage'] = 'down'
                        
                        form_warning = ""
                        if asym > AR_METRICS['max_asymmetry']:
                            form_warning = "⚠ Raise both arms evenly to overhead position"
                        elif max_elev > AR_METRICS['max_shoulder_elevation']:
                            form_warning = "⚠ Avoid shrugging trapezius muscles at top of lift"
                        ar_state['form_warning'] = form_warning
                except Exception:
                    pass
            
            draw_studio_skeleton(
                image,
                results,
                joint_badges=joint_badges,
                has_warning=bool(ar_state.get('form_warning'))
            )
            
            ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()


# ============================================================================
# 4. MODIFIED JUMPING JACKS (JJ) - METRICS, STATE & STREAMING
# ============================================================================

JJ_METRICS = {
    "arm_down_threshold": 45,
    "arm_up_threshold": 135,
    "excellent_arm_peak": 150,
    "stance_out_ratio": 1.45,
    "excellent_stance_ratio": 1.55,
    "stance_in_ratio": 1.25,
}

def _default_jj_rep():
    return {
        "max_arm_angle": 0.0,
        "min_arm_angle": 180.0,
        "max_stance_ratio": 1.0,
        "max_asymmetry": 0.0,
    }

jj_state = {
    'exercise_type': 'Modified Jumping Jacks',
    'active': False,
    'complete': False,
    'counter': 0,
    'target_reps': 12,
    'stage': None,
    'excellent_reps': 0,
    'good_reps': 0,
    'partial_reps': 0,
    'current_rep': _default_jj_rep(),
    'left_shoulder_angle': 0,
    'right_shoulder_angle': 0,
    'stance_ratio': 1.0,
    'symmetry_score': 100,
    'stance_score': 50,
    'form_warning': '',
    'start_time': None,
    'workout_record': None,
}


def generate_jj_frames():
    """Generate clean studio video frames for Modified Jumping Jacks (Overhead Abduction + Ankle Step-Out)"""
    global jj_state
    
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            joint_badges = []
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    l_sh = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    l_wr = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
                    l_hp = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
                    l_ak_lm = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    
                    r_sh = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    r_wr = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
                    r_hp = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
                    r_ak_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    
                    # Overhead arm abduction angle (Hip - Shoulder - Wrist)
                    l_arm_angle = calculate_angle(l_hp, l_sh, l_wr)
                    r_arm_angle = calculate_angle(r_hp, r_sh, r_wr)
                    avg_arm_angle = (l_arm_angle + r_arm_angle) / 2.0
                    asym = abs(l_arm_angle - r_arm_angle)
                    
                    jj_state['left_shoulder_angle'] = l_arm_angle
                    jj_state['right_shoulder_angle'] = r_arm_angle
                    jj_state['symmetry_score'] = max(0, min(100, int(100 - (asym / 25.0) * 100)))
                    
                    # Lower-body lateral step-out: Euclidean ankle distance normalized by hip width
                    hip_width = max(0.05, float(np.linalg.norm(np.array(l_hp) - np.array(r_hp))))
                    ankles_visible = (l_ak_lm.visibility > 0.45 and r_ak_lm.visibility > 0.45)
                    if ankles_visible:
                        l_ak = [l_ak_lm.x, l_ak_lm.y]
                        r_ak = [r_ak_lm.x, r_ak_lm.y]
                        ankle_dist = float(np.linalg.norm(np.array(l_ak) - np.array(r_ak)))
                        stance_ratio = round(ankle_dist / hip_width, 2)
                    else:
                        # Fallback estimate from arm abduction when lower body is out of webcam frame
                        stance_ratio = round(1.0 + max(0.0, (avg_arm_angle - 40.0) / 150.0) * 0.65, 2)
                    
                    jj_state['stance_ratio'] = stance_ratio
                    jj_state['stance_score'] = max(0, min(100, int(((stance_ratio - 1.0) / 0.7) * 100)))
                    
                    joint_badges = [
                        (l_sh, f"L {int(l_arm_angle)} deg"),
                        (r_sh, f"R {int(r_arm_angle)} deg"),
                    ]
                    
                    if jj_state['active'] and not jj_state['complete']:
                        cur = jj_state['current_rep']
                        cur['max_arm_angle'] = max(cur['max_arm_angle'], avg_arm_angle)
                        cur['min_arm_angle'] = min(cur['min_arm_angle'], avg_arm_angle)
                        cur['max_stance_ratio'] = max(cur['max_stance_ratio'], stance_ratio)
                        cur['max_asymmetry'] = max(cur['max_asymmetry'], asym)
                        
                        arms_up = avg_arm_angle >= JJ_METRICS['arm_up_threshold']
                        stance_out = (not ankles_visible) or (stance_ratio >= JJ_METRICS['stance_out_ratio'])
                        
                        if arms_up and stance_out and jj_state['stage'] != 'up':
                            jj_state['stage'] = 'up'
                        elif avg_arm_angle <= JJ_METRICS['arm_down_threshold'] and jj_state['stage'] == 'up':
                            rom = cur['max_arm_angle'] - cur['min_arm_angle']
                            if (cur['max_arm_angle'] >= JJ_METRICS['excellent_arm_peak'] and
                                cur['max_stance_ratio'] >= JJ_METRICS['excellent_stance_ratio'] and
                                cur['max_asymmetry'] <= 15):
                                jj_state['excellent_reps'] += 1
                                jj_state['good_reps'] += 1
                            elif cur['max_arm_angle'] >= JJ_METRICS['arm_up_threshold'] and rom >= 80:
                                jj_state['good_reps'] += 1
                            else:
                                jj_state['partial_reps'] += 1
                            
                            jj_state['stage'] = 'down'
                            jj_state['counter'] += 1
                            jj_state['current_rep'] = _default_jj_rep()
                            
                            if jj_state['counter'] >= jj_state['target_reps']:
                                jj_state['complete'] = True
                                jj_state['active'] = False
                        elif jj_state['stage'] is None and avg_arm_angle <= JJ_METRICS['arm_down_threshold']:
                            jj_state['stage'] = 'down'
                        
                        form_warning = ""
                        if arms_up and ankles_visible and stance_ratio < JJ_METRICS['stance_out_ratio']:
                            form_warning = "⚠ Step feet wider (>1.5x hip width) with overhead reach"
                        elif asym > 18:
                            form_warning = "⚠ Keep both arms symmetrical during overhead abduction"
                        jj_state['form_warning'] = form_warning
                except Exception:
                    pass
            
            draw_studio_skeleton(
                image,
                results,
                joint_badges=joint_badges,
                has_warning=bool(jj_state.get('form_warning'))
            )
            
            ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()


# ============================================================================
# WORKOUT CONTROL & TELEMETRY API VIEWS (ALL EXERCISES)
# ============================================================================

def _sync_state_record(state_dict):
    """Persist live progress to the active SLS database record if present"""
    record = state_dict.get('workout_record')
    if record:
        try:
            record.total_reps = state_dict['counter']
            record.excellent_reps = state_dict['excellent_reps']
            record.good_reps = state_dict['good_reps']
            record.partial_reps = state_dict['partial_reps']
            record.completed = state_dict['complete']
            if state_dict.get('start_time'):
                record.duration_seconds = int(time.time() - state_dict['start_time'])
            record.save()
        except Exception as e:
            print(f"Error syncing workout record: {e}")


def _create_workout_record(request, state_dict, exercise_name, target_reps):
    """Create initial SLS database record and log workout start to Patient Logs"""
    patient_id = request.session.get('patient_id')
    if patient_id:
        try:
            patient = Patient.objects.get(id=patient_id)
            state_dict['workout_record'] = SLS.objects.create(
                patient=patient,
                exercise_type=exercise_name,
                target_reps=target_reps,
                date=timezone.now().date()
            )
            workout_log = f"""{exercise_name.upper()} - WORKOUT STARTED
Exercise: {exercise_name}
Target Reps: {target_reps}
Start Time: {timezone.now().strftime('%H:%M:%S')}
"""
            append_to_patient_log(patient, workout_log)
        except Patient.DoesNotExist:
            state_dict['workout_record'] = None
    else:
        state_dict['workout_record'] = None


# --- SIDE LATERAL RAISE API ---

@csrf_exempt
def start_workout(request):
    """Start the Side Lateral Raise workout and create database record"""
    global workout_state
    if request.method == 'POST':
        data = json.loads(request.body)
        target_reps = int(data.get('target_reps', 12))
        
        workout_state['active'] = True
        workout_state['complete'] = False
        workout_state['counter'] = 0
        workout_state['stage'] = None
        workout_state['excellent_reps'] = 0
        workout_state['good_reps'] = 0
        workout_state['partial_reps'] = 0
        workout_state['target_reps'] = target_reps
        workout_state['initial_left_shoulder'] = None
        workout_state['initial_right_shoulder'] = None
        workout_state['form_warning'] = ''
        workout_state['start_time'] = time.time()
        workout_state['current_rep'] = _default_sr_rep()
        
        _create_workout_record(request, workout_state, 'Side Lateral Raise', target_reps)
        return JsonResponse({'status': 'started', 'target_reps': target_reps})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def reset_workout(request):
    """Reset the Side Lateral Raise workout state"""
    global workout_state
    if request.method == 'POST':
        _sync_state_record(workout_state)
        workout_state['active'] = False
        workout_state['complete'] = False
        workout_state['counter'] = 0
        workout_state['stage'] = None
        workout_state['excellent_reps'] = 0
        workout_state['good_reps'] = 0
        workout_state['partial_reps'] = 0
        workout_state['initial_left_shoulder'] = None
        workout_state['initial_right_shoulder'] = None
        workout_state['left_shoulder_angle'] = 0
        workout_state['right_shoulder_angle'] = 0
        workout_state['asymmetry'] = 0
        workout_state['shrug_level'] = 0
        workout_state['symmetry_score'] = 100
        workout_state['shrug_score'] = 100
        workout_state['form_warning'] = ""
        workout_state['current_rep'] = _default_sr_rep()
        workout_state['workout_record'] = None
        workout_state['start_time'] = None
        return JsonResponse({'status': 'reset'})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def get_workout_status(request):
    """Get current Side Lateral Raise workout telemetry and sync DB"""
    global workout_state
    _sync_state_record(workout_state)
    return JsonResponse({
        'counter': workout_state['counter'],
        'stage': workout_state['stage'],
        'excellent_reps': workout_state['excellent_reps'],
        'good_reps': workout_state['good_reps'],
        'partial_reps': workout_state['partial_reps'],
        'target_reps': workout_state['target_reps'],
        'complete': workout_state['complete'],
        'active': workout_state['active'],
        'left_shoulder_angle': int(workout_state['left_shoulder_angle']),
        'right_shoulder_angle': int(workout_state['right_shoulder_angle']),
        'left_elbow_angle': int(workout_state['left_elbow_angle']),
        'right_elbow_angle': int(workout_state['right_elbow_angle']),
        'asymmetry': int(workout_state['asymmetry']),
        'symmetry_score': workout_state.get('symmetry_score', 100),
        'shrug_score': workout_state.get('shrug_score', 100),
        'form_warning': workout_state['form_warning']
    })


# --- BICEP CURLS API ---

@csrf_exempt
def start_bc_workout(request):
    """Start Bicep Curls workout"""
    global bc_state
    if request.method == 'POST':
        data = json.loads(request.body)
        target_reps = int(data.get('target_reps', 10))
        arm_mode = data.get('arm_mode', 'BILATERAL')  # 'BILATERAL' or 'SEQUENTIAL'
        
        bc_state['active'] = True
        bc_state['complete'] = False
        bc_state['counter'] = 0
        bc_state['target_reps'] = target_reps
        bc_state['arm_mode'] = arm_mode
        bc_state['current_arm'] = 'LEFT' if arm_mode == 'SEQUENTIAL' else 'BOTH'
        bc_state['left_counter'] = 0
        bc_state['right_counter'] = 0
        bc_state['left_stage'] = None
        bc_state['right_stage'] = None
        bc_state['left_initial_shoulder'] = None
        bc_state['right_initial_shoulder'] = None
        bc_state['left_current_rep'] = _default_bc_arm_rep()
        bc_state['right_current_rep'] = _default_bc_arm_rep()
        bc_state['excellent_reps'] = 0
        bc_state['good_reps'] = 0
        bc_state['partial_reps'] = 0
        bc_state['form_warning'] = ''
        bc_state['start_time'] = time.time()
        
        _create_workout_record(request, bc_state, 'Bicep Curls', target_reps)
        return JsonResponse({'status': 'started', 'target_reps': target_reps, 'current_arm': bc_state['current_arm']})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def switch_bc_arm(request):
    """Switch active arm in Bicep Curls tracker"""
    global bc_state
    if request.method == 'POST':
        data = json.loads(request.body) if request.body else {}
        requested_arm = data.get('arm')
        if requested_arm in ('LEFT', 'RIGHT', 'BOTH'):
            bc_state['current_arm'] = requested_arm
        else:
            bc_state['current_arm'] = 'RIGHT' if bc_state['current_arm'] == 'LEFT' else 'LEFT'
        bc_state['form_warning'] = ''
        return JsonResponse({'status': 'switched', 'current_arm': bc_state['current_arm']})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def reset_bc_workout(request):
    """Reset Bicep Curls workout"""
    global bc_state
    if request.method == 'POST':
        _sync_state_record(bc_state)
        bc_state['active'] = False
        bc_state['complete'] = False
        bc_state['counter'] = 0
        bc_state['left_counter'] = 0
        bc_state['right_counter'] = 0
        bc_state['left_stage'] = None
        bc_state['right_stage'] = None
        bc_state['left_initial_shoulder'] = None
        bc_state['right_initial_shoulder'] = None
        bc_state['left_current_rep'] = _default_bc_arm_rep()
        bc_state['right_current_rep'] = _default_bc_arm_rep()
        bc_state['excellent_reps'] = 0
        bc_state['good_reps'] = 0
        bc_state['partial_reps'] = 0
        bc_state['stability_score'] = 100
        bc_state['rom_score'] = 0
        bc_state['form_warning'] = ''
        bc_state['workout_record'] = None
        bc_state['start_time'] = None
        return JsonResponse({'status': 'reset'})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def get_bc_status(request):
    """Get Bicep Curls live telemetry"""
    global bc_state
    _sync_state_record(bc_state)
    return JsonResponse({
        'counter': bc_state['counter'],
        'left_counter': bc_state['left_counter'],
        'right_counter': bc_state['right_counter'],
        'current_arm': bc_state['current_arm'],
        'stage': bc_state['stage'],
        'excellent_reps': bc_state['excellent_reps'],
        'good_reps': bc_state['good_reps'],
        'partial_reps': bc_state['partial_reps'],
        'target_reps': bc_state['target_reps'],
        'complete': bc_state['complete'],
        'active': bc_state['active'],
        'left_angle': int(bc_state['left_elbow_angle']),
        'right_angle': int(bc_state['right_elbow_angle']),
        'left_drift': bc_state['left_drift'],
        'right_drift': bc_state['right_drift'],
        'stability_score': bc_state.get('stability_score', 100),
        'rom_score': bc_state.get('rom_score', 0),
        'form_warning': bc_state['form_warning']
    })


# --- ARM RAISES API ---

@csrf_exempt
def start_ar_workout(request):
    """Start Arm Raises workout"""
    global ar_state
    if request.method == 'POST':
        data = json.loads(request.body)
        target_reps = int(data.get('target_reps', 10))
        
        ar_state['active'] = True
        ar_state['complete'] = False
        ar_state['counter'] = 0
        ar_state['target_reps'] = target_reps
        ar_state['stage'] = None
        ar_state['excellent_reps'] = 0
        ar_state['good_reps'] = 0
        ar_state['partial_reps'] = 0
        ar_state['current_rep'] = _default_ar_rep()
        ar_state['initial_left_shoulder'] = None
        ar_state['initial_right_shoulder'] = None
        ar_state['form_warning'] = ''
        ar_state['start_time'] = time.time()
        
        _create_workout_record(request, ar_state, 'Arm Raises', target_reps)
        return JsonResponse({'status': 'started', 'target_reps': target_reps})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def reset_ar_workout(request):
    """Reset Arm Raises workout"""
    global ar_state
    if request.method == 'POST':
        _sync_state_record(ar_state)
        ar_state['active'] = False
        ar_state['complete'] = False
        ar_state['counter'] = 0
        ar_state['stage'] = None
        ar_state['excellent_reps'] = 0
        ar_state['good_reps'] = 0
        ar_state['partial_reps'] = 0
        ar_state['current_rep'] = _default_ar_rep()
        ar_state['initial_left_shoulder'] = None
        ar_state['initial_right_shoulder'] = None
        ar_state['symmetry_score'] = 100
        ar_state['shrug_score'] = 100
        ar_state['form_warning'] = ''
        ar_state['workout_record'] = None
        ar_state['start_time'] = None
        return JsonResponse({'status': 'reset'})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def get_ar_status(request):
    """Get Arm Raises live telemetry"""
    global ar_state
    _sync_state_record(ar_state)
    return JsonResponse({
        'counter': ar_state['counter'],
        'stage': ar_state['stage'],
        'excellent_reps': ar_state['excellent_reps'],
        'good_reps': ar_state['good_reps'],
        'partial_reps': ar_state['partial_reps'],
        'target_reps': ar_state['target_reps'],
        'complete': ar_state['complete'],
        'active': ar_state['active'],
        'left_angle': int(ar_state['left_shoulder_angle']),
        'right_angle': int(ar_state['right_shoulder_angle']),
        'asymmetry': int(ar_state['asymmetry']),
        'symmetry_score': ar_state.get('symmetry_score', 100),
        'shrug_score': ar_state.get('shrug_score', 100),
        'form_warning': ar_state['form_warning']
    })


# --- MODIFIED JUMPING JACKS API ---

@csrf_exempt
def start_jj_workout(request):
    """Start Modified Jumping Jacks workout"""
    global jj_state
    if request.method == 'POST':
        data = json.loads(request.body)
        target_reps = int(data.get('target_reps', 12))
        
        jj_state['active'] = True
        jj_state['complete'] = False
        jj_state['counter'] = 0
        jj_state['target_reps'] = target_reps
        jj_state['stage'] = None
        jj_state['excellent_reps'] = 0
        jj_state['good_reps'] = 0
        jj_state['partial_reps'] = 0
        jj_state['current_rep'] = _default_jj_rep()
        jj_state['form_warning'] = ''
        jj_state['start_time'] = time.time()
        
        _create_workout_record(request, jj_state, 'Modified Jumping Jacks', target_reps)
        return JsonResponse({'status': 'started', 'target_reps': target_reps})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def reset_jj_workout(request):
    """Reset Modified Jumping Jacks workout"""
    global jj_state
    if request.method == 'POST':
        _sync_state_record(jj_state)
        jj_state['active'] = False
        jj_state['complete'] = False
        jj_state['counter'] = 0
        jj_state['stage'] = None
        jj_state['excellent_reps'] = 0
        jj_state['good_reps'] = 0
        jj_state['partial_reps'] = 0
        jj_state['current_rep'] = _default_jj_rep()
        jj_state['symmetry_score'] = 100
        jj_state['stance_score'] = 50
        jj_state['form_warning'] = ''
        jj_state['workout_record'] = None
        jj_state['start_time'] = None
        return JsonResponse({'status': 'reset'})
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def get_jj_status(request):
    """Get Modified Jumping Jacks live telemetry"""
    global jj_state
    _sync_state_record(jj_state)
    return JsonResponse({
        'counter': jj_state['counter'],
        'stage': jj_state['stage'],
        'excellent_reps': jj_state['excellent_reps'],
        'good_reps': jj_state['good_reps'],
        'partial_reps': jj_state['partial_reps'],
        'target_reps': jj_state['target_reps'],
        'complete': jj_state['complete'],
        'active': jj_state['active'],
        'left_angle': int(jj_state['left_shoulder_angle']),
        'right_angle': int(jj_state['right_shoulder_angle']),
        'stance_ratio': jj_state['stance_ratio'],
        'symmetry_score': jj_state.get('symmetry_score', 100),
        'stance_score': jj_state.get('stance_score', 50),
        'form_warning': jj_state['form_warning']
    })


# --- UNIFIED WORKOUT PERSISTENCE & AUDIT LOGGING ---

@csrf_exempt
def save_workout(request):
    """Save completed workout for any exercise (SR, BC, AR, JJ) to database and patient medical log"""
    if request.method == 'POST':
        try:
            patient_id = request.session.get('patient_id')
            data = json.loads(request.body)
            exercise_type = data.get('exercise_type', 'Side Lateral Raise')
            
            if not patient_id:
                # Allow guest demo mode to return graceful response while noting session status
                return JsonResponse({
                    'status': 'success',
                    'workout_id': None,
                    'message': 'Session completed (Log in as patient to persist to medical record).'
                })
            
            patient = Patient.objects.get(id=patient_id)
            
            # Map exercise_type to its state dictionary so we reuse the active workout_record if one was created at start
            state_map = {
                'Side Lateral Raise': workout_state,
                'Bicep Curls': bc_state,
                'Arm Raises': ar_state,
                'Modified Jumping Jacks': jj_state,
            }
            active_state = state_map.get(exercise_type)
            existing_record = active_state.get('workout_record') if active_state else None
            
            if existing_record:
                workout = existing_record
                workout.exercise_type = exercise_type
                workout.total_reps = data.get('total_reps', 0)
                workout.target_reps = data.get('target_reps', 10)
                workout.excellent_reps = data.get('excellent_reps', 0)
                workout.good_reps = data.get('good_reps', 0)
                workout.partial_reps = data.get('partial_reps', 0)
                workout.duration_seconds = data.get('duration_seconds', 0)
                workout.completed = data.get('completed', False)
                workout.save()
                if active_state:
                    active_state['workout_record'] = None
            else:
                workout = SLS.objects.create(
                    patient=patient,
                    exercise_type=exercise_type,
                    date=date.today(),
                    total_reps=data.get('total_reps', 0),
                    target_reps=data.get('target_reps', 10),
                    excellent_reps=data.get('excellent_reps', 0),
                    good_reps=data.get('good_reps', 0),
                    partial_reps=data.get('partial_reps', 0),
                    duration_seconds=data.get('duration_seconds', 0),
                    completed=data.get('completed', False)
                )
            
            duration_min = workout.duration_seconds // 60
            duration_sec = workout.duration_seconds % 60
            good_only = max(0, workout.good_reps - workout.excellent_reps)
            
            workout_log = f"""{exercise_type.upper()} - WORKOUT COMPLETED
Exercise: {exercise_type}
Date: {workout.date}
Duration: {duration_min}m {duration_sec}s

RESULTS:
- Total Reps: {workout.total_reps}/{workout.target_reps}
- Excellent Reps (★): {workout.excellent_reps}
- Good Reps (✓): {good_only}
- Partial Reps: {workout.partial_reps}
- Completion: {workout.completion_percentage}%
- Quality Score: {workout.quality_score}/100

Status: {'✅ COMPLETED' if workout.completed else '⚠️ INCOMPLETE'}
"""
            append_to_patient_log(patient, workout_log)
            
            return JsonResponse({
                'status': 'success',
                'workout_id': workout.id,
                'quality_score': workout.quality_score,
                'message': f'{exercise_type} workout saved to clinical record!'
            })
            
        except Patient.DoesNotExist:
            return JsonResponse({'error': 'Patient not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


# ============================================================================
# EXERCISE PAGE & VIDEO FEED VIEWS
# ============================================================================

def exercise(request):
    patient = None
    custom_exercises = []
    patient_id = request.session.get('patient_id')
    if patient_id:
        patient = Patient.objects.filter(id=patient_id).select_related('doctor').first()
        if patient:
            custom_exercises = list(
                CustomExercise.objects.filter(patient=patient)
                .select_related('doctor')
                .order_by('-created_at')
            )
    return render(request, 'exercise.html', {
        'patient': patient,
        'custom_exercises': custom_exercises,
    })


def bc(request):
    return render(request, 'bc.html')


def ar(request):
    return render(request, 'ar.html')


def jj(request):
    return render(request, 'jj.html')


def srtwo(request):
    """Side Lateral Raise tracker main view"""
    return render(request, 'side_lateral_raise.html')


def video_feed(request):
    """Video streaming route for Side Lateral Raise"""
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def bc_video_feed(request):
    """Video streaming route for Bicep Curls"""
    return StreamingHttpResponse(
        generate_bc_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def ar_video_feed(request):
    """Video streaming route for Arm Raises"""
    return StreamingHttpResponse(
        generate_ar_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def jj_video_feed(request):
    """Video streaming route for Modified Jumping Jacks"""
    return StreamingHttpResponse(
        generate_jj_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def sr(request):
    """Side Lateral Raise dashboard with workout history"""
    patient_id = request.session.get('patient_id')
    
    if not patient_id:
        messages.error(request, 'Please log in to view your workout history.')
        return redirect('login')
    
    try:
        patient = Patient.objects.get(id=patient_id)
        sls_workouts = SLS.objects.filter(patient=patient).order_by('date', 'created_at')
        
        workout_data = []
        for workout in list(sls_workouts)[-10:]:
            good_only = max(0, workout.good_reps - workout.excellent_reps)
            workout_data.append({
                'date': workout.date.isoformat(),
                'exercise_type': getattr(workout, 'exercise_type', 'Side Lateral Raise'),
                'excellent_reps': workout.excellent_reps,
                'good_reps': good_only,
                'partial_reps': workout.partial_reps,
                'total_reps': workout.total_reps,
                'completion_percentage': workout.completion_percentage,
                'quality_score': workout.quality_score
            })
        
        if not workout_data:
            workout_data = [
                {
                    'date': (date.today() - timedelta(days=1)).isoformat(),
                    'exercise_type': 'Side Lateral Raise',
                    'excellent_reps': 0,
                    'good_reps': 0,
                    'partial_reps': 0,
                    'total_reps': 0,
                    'completion_percentage': 0,
                    'quality_score': 0
                }
            ]
        
        context = {
            'patient': patient,
            'workout_data': json.dumps(workout_data),
        }
        return render(request, 'sr.html', context)
        
    except Patient.DoesNotExist:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('login')


# ============================================================================
# CLINICAL LOGGING & AI SUMMARY GENERATION
# ============================================================================

def append_to_patient_log(patient, log_entry):
    """
    Append information to patient's log.
    Creates a log if it doesn't exist, or appends to existing log.
    """
    try:
        log, created = Logs.objects.get_or_create(
            patient=patient,
            defaults={
                'name': patient.name,
                'email': patient.email,
                'text': ''
            }
        )
        
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry_formatted = f"\n{'='*80}\n[{timestamp}]\n{log_entry}\n{'='*80}\n"
        log.text += log_entry_formatted
        log.save()
        
        print(f"[OK] Log updated for {patient.name}")
        
    except Exception as e:
        print(f"[WARN] Failed to update log: {str(e)}")


@require_http_methods(["POST"])
def generate_and_send_summary(request):
    patient_id = request.session.get("patient_id")
    if not patient_id:
        return JsonResponse({"success": False, "error": "Not authenticated"})

    try:
        patient = Patient.objects.get(id=patient_id)
        log = Logs.objects.get(patient=patient)

        llm = initialize_model()

        system_prompt = """
You are a clinical rehabilitation assistant.
Create a concise, structured medical summary for a doctor.
The summary must be easy to scan and medically useful.

Include:
• Patient condition & symptoms
• Rehab progress
• Pain / difficulty patterns
• Risks or red flags (if any)

Avoid casual language. Be professional.
"""

        human_prompt = f"""
PATIENT INFO:
{patient.info}

CHAT LOGS:
{log.text}

"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        summary = response.content.strip()

        # ---- STORE SUMMARY (REPLACE) ----
        patient.summary = summary
        patient.updated_at = timezone.now()
        patient.save(update_fields=["summary", "updated_at"])

        # ---- EMAIL (same subject = same Gmail thread) ----
        subject = f"Patient Summary – {patient.name}"

        return JsonResponse({
            "success": True,
            "message": "Summary generated, stored, and emailed",
            "summary": summary,
        })

    except Logs.DoesNotExist:
        return JsonResponse({"success": False, "error": "No logs found"})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})
