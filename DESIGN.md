# ReviveCare - System Design Document

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Models](#data-models)
4. [Core Features](#core-features)
5. [AI Integration](#ai-integration)
6. [Exercise Tracking System](#exercise-tracking-system)
7. [Security & Authentication](#security--authentication)
8. [User Flows](#user-flows)
9. [Technical Decisions](#technical-decisions)
10. [Future Enhancements](#future-enhancements)

---

## System Overview

### Purpose
ReviveCare is a post-discharge recovery support platform that provides continuous monitoring, AI-powered guidance, and real-time exercise tracking for patients recovering from surgery or injury.

### Key Objectives
- Bridge the gap between hospital discharge and full recovery
- Provide 24/7 AI-powered health assistance
- Enable real-time exercise form correction
- Automate doctor notifications for concerning symptoms
- Track and visualize recovery progress

### Target Users
- **Patients**: Individuals recovering from surgery or injury at home
- **Doctors**: Healthcare providers monitoring patient recovery

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Browser    │  │   Webcam     │  │   Mobile     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Django Web Framework                     │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │   │
│  │  │  Home  │  │Patient │  │ Doctor │  │ Admin  │    │   │
│  │  │  App   │  │  App   │  │  App   │  │ Panel  │    │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   AI     │  │  Pose    │  │  Email   │  │  Twilio  │   │
│  │ Chatbot  │  │Detection │  │  SMTP    │  │  Voice   │   │
│  │(LangChain│  │(MediaPipe│  │          │  │          │   │
│  │  +Groq)  │  │ +OpenCV) │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  SQLite Database                      │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                 │   │
│  │  │Patient │  │  SLS   │  │  Logs  │                 │   │
│  │  │ Model  │  │ Model  │  │ Model  │                 │   │
│  │  └────────┘  └────────┘  └────────┘                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend**
- Django 5.2.1 (Web framework)
- Python 3.8+ (Programming language)
- SQLite (Database)

**AI/ML**
- LangChain (LLM orchestration)
- Groq (LLM provider - Llama 3.3 70B)
- MediaPipe (Pose detection)
- OpenCV (Computer vision)
- NumPy (Numerical computations)

**Frontend**
- HTML5, CSS3, JavaScript
- Responsive design (mobile-first)
- Real-time video streaming

**External Services**
- Twilio (Voice calls for emergencies)
- SMTP (Email notifications)

---

## Data Models

### Patient Model
```python
class Patient(models.Model):
    name = CharField(max_length=100)
    email = EmailField(unique=True)  # Primary identifier
    info = TextField(max_length=3000)  # Medical context
    summary = TextField(max_length=5000, null=True, blank=True)  # AI-generated
    created_at = DateTimeField(default=timezone.now)
    updated_at = DateTimeField(auto_now=True)
    is_active = BooleanField(default=True)
```

**Design Rationale**:
- Email as unique identifier (no password authentication)
- `info` field stores doctor-provided medical context for AI chatbot
- `summary` field stores AI-generated recovery insights
- Timestamps for audit trail

### SLS (Side Lateral Raise) Model
```python
class SLS(models.Model):
    patient = ForeignKey(Patient, on_delete=CASCADE)
    date = DateField(default=timezone.now)
    
    # Workout metrics
    total_reps = IntegerField(default=0)
    target_reps = IntegerField(default=10)
    excellent_reps = IntegerField(default=0)
    good_reps = IntegerField(default=0)
    partial_reps = IntegerField(default=0)
    
    # Session details
    duration_seconds = IntegerField(default=0)
    completed = BooleanField(default=False)
    
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
```

**Design Rationale**:
- Tracks exercise quality (excellent/good/partial)
- Stores duration for progress analysis
- Completion flag for workout adherence tracking
- Computed properties: `completion_percentage`, `quality_score`

### Logs Model
```python
class Logs(models.Model):
    patient = ForeignKey(Patient, on_delete=CASCADE)
    name = CharField(max_length=1000)
    email = EmailField()
    text = TextField(max_length=100000, default="")  # Append-only log
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
```

**Design Rationale**:
- Append-only design for complete activity history
- Stores chat conversations, exercise logs, and system events
- Used for AI summary generation
- Timestamped entries for chronological analysis

---

## Core Features

### 1. Patient Authentication

**Design**: Email-only authentication (no passwords)

**Flow**:
1. Patient enters email
2. System checks if email exists in database
3. If found: Create session, redirect to dashboard
4. If not found: Show "Doctor hasn't added you yet" message

**Rationale**:
- Simplifies onboarding (no password management)
- Doctor controls access by adding patient emails
- Reduces friction for elderly or non-tech-savvy users

**Session Management**:
```python
request.session['patient_id'] = patient.id
request.session['patient_email'] = patient.email
request.session['patient_name'] = patient.name
```

### 2. AI Chatbot System

**Architecture**:
```
User Query → LangChain → Groq (Llama 3.3) → Response Parser → JSON Output
                ↓
         Patient Context
         (from database)
```

**System Prompt Design**:
- Role: Post-discharge recovery support chatbot
- Constraints: Cannot diagnose, prescribe, or alter treatment
- Input: Patient medical context + conversation history
- Output: JSON with `patient_response` and `seriousness_score`

**Seriousness Scoring**:
- 0.00-0.30: Normal recovery, reassurance sufficient
- 0.31-0.60: Monitor closely, routine follow-up
- 0.61-0.75: Concerning, contact doctor soon
- >0.75: Potential emergency, immediate escalation

**Emergency Alert Flow**:
```
Seriousness > 0.75 → Email to Doctor + Twilio Voice Call
```

**Conversation History**:
- Stored in session (last 20 messages)
- Logged to database for summary generation
- Maintains context across chat session

### 3. Exercise Tracking System

**Pose Detection Pipeline**:
```
Webcam → OpenCV → MediaPipe Pose → Landmark Extraction → Angle Calculation → Rep Counting → Quality Assessment
```

**Real-time Processing**:
1. Capture video frame
2. Detect 33 body landmarks
3. Calculate joint angles
4. Track rep stages (up/down)
5. Assess form quality
6. Display feedback overlay

**Quality Metrics** (Side Lateral Raise):
- **Excellent**: Peak angle ≥105°, ROM ≥60°, elbow straight, no shrugging, symmetric
- **Good**: Peak angle ≥95°, ROM ≥60°, reasonably symmetric
- **Partial**: Completed but below quality thresholds

**Form Warnings**:
- Asymmetry >15° between arms
- Shoulder elevation (shrugging)
- Bent elbows (<165°)

### 4. Recovery Insights

**AI Summary Generation**:
```
Patient Info + Chat Logs + Exercise Data → LLM → Structured Summary → Email to Doctor
```

**Summary Contents**:
- Patient condition and symptoms
- Rehabilitation progress
- Pain/difficulty patterns
- Risk factors or red flags
- Professional medical tone

**Storage**:
- Stored in `Patient.summary` field
- Replaces previous summary (not append)
- Emailed to doctor with consistent subject line (Gmail threading)

---

## AI Integration

### LangChain Configuration

```python
ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,  # Low for consistent medical advice
    max_tokens=1024,
    timeout=30,
    max_retries=2
)
```

**Model Selection Rationale**:
- Llama 3.3 70B: Strong reasoning for medical context
- Low temperature: Consistent, reliable responses
- Groq: Fast inference for real-time chat

### Prompt Engineering

**Chatbot System Prompt**:
- Clear role definition
- Explicit constraints (no diagnosis/prescription)
- Structured output format (JSON)
- Scoring guidelines with examples
- Patient-specific context injection

**Summary System Prompt**:
- Professional medical tone
- Structured format for doctor review
- Focus on actionable insights
- Concise and scannable

### Context Management

**Patient Context Injection**:
```python
system_prompt = SYSTEM_PROMPT.format(patient_context=patient.info)
```

**Conversation History**:
```python
messages = [SystemMessage, ...history..., HumanMessage]
```

---

## Exercise Tracking System

### Pose Detection Configuration

**MediaPipe Settings**:
```python
mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Angle Calculation

```python
def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c"""
    radians = arctan2(c[1]-b[1], c[0]-b[0]) - arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
```

### Rep Detection State Machine

```
State: None → down → up → down (rep counted)
```

**Transitions**:
- `None → down`: Initial position
- `down → up`: Shoulder angle > up_threshold (100°)
- `up → down`: Shoulder angle < down_threshold (35°) + quality assessment

### Workout State Management

**Global State** (thread-safe for single user):
```python
workout_state = {
    'active': bool,
    'complete': bool,
    'counter': int,
    'stage': str,
    'excellent_reps': int,
    'good_reps': int,
    'partial_reps': int,
    'target_reps': int,
    'current_rep': dict,  # Metrics for ongoing rep
    'form_warning': str
}
```

### Video Streaming

**Technology**: HTTP multipart streaming
```python
StreamingHttpResponse(
    generate_frames(),
    content_type='multipart/x-mixed-replace; boundary=frame'
)
```

**Frame Generation**:
1. Capture frame from webcam
2. Process with MediaPipe
3. Draw overlays (angles, rep count, warnings)
4. Encode as JPEG
5. Yield as multipart response

---

## Security & Authentication

### Authentication Strategy

**Email-Only Authentication**:
- No passwords stored
- Doctor controls access
- Session-based authorization

**Session Security**:
- Django session framework
- CSRF protection enabled
- Session timeout on browser close

### Data Protection

**Environment Variables**:
```
GROQ_API_KEY
EMAIL_PASSWORD
TWILIO_AUTH_TOKEN
```

**Sensitive Data**:
- Medical information in `Patient.info`
- Chat logs in `Logs.text`
- No PII in logs or error messages

### Production Recommendations

1. **HTTPS**: Mandatory for production
2. **Database**: Migrate to PostgreSQL
3. **Secrets Management**: Use AWS Secrets Manager or similar
4. **Rate Limiting**: Implement for API endpoints
5. **Input Validation**: Sanitize all user inputs
6. **CORS**: Configure for specific domains

---

## User Flows

### Patient Journey

```
1. Doctor adds patient email + medical context
2. Patient receives notification (manual or automated)
3. Patient visits site, enters email
4. System authenticates, creates session
5. Patient dashboard loads with personalized data
6. Patient can:
   - Chat with AI assistant
   - Perform guided exercises
   - View recovery insights
   - Track progress
7. System logs all activities
8. AI monitors for concerning symptoms
9. Doctor receives automated summaries
```

### Doctor Workflow

```
1. Access doctor portal
2. Enter patient information:
   - Name
   - Email (unique identifier)
   - Medical context (surgery, medications, restrictions)
3. System creates/updates patient record
4. Patient can now log in
5. Doctor receives:
   - Emergency alerts (seriousness > 0.75)
   - Automated summaries (on-demand)
6. Doctor can update patient info as recovery progresses
```

### Exercise Session Flow

```
1. Patient navigates to exercise page
2. Selects exercise type (e.g., Side Lateral Raise)
3. Sets target reps
4. Clicks "Start Workout"
5. Webcam activates, pose detection begins
6. Real-time feedback displayed:
   - Rep count
   - Joint angles
   - Form warnings
   - Quality assessment
7. Workout completes when target reached
8. Results saved to database
9. Log entry created
10. Dashboard updated with new data
```

---

## Technical Decisions

### Why Django?

**Pros**:
- Rapid development with built-in admin, ORM, auth
- Excellent documentation
- Strong ecosystem for ML integration
- Session management out-of-the-box

**Cons**:
- Monolithic (could use microservices for scale)
- Synchronous by default (async needed for real-time features)

### Why SQLite?

**Pros**:
- Zero configuration
- Perfect for MVP/prototype
- File-based (easy backup)

**Cons**:
- Not suitable for production scale
- No concurrent writes
- Limited for analytics queries

**Migration Path**: PostgreSQL for production

### Why Email-Only Auth?

**Pros**:
- Simplifies onboarding
- No password reset flows
- Doctor-controlled access
- Suitable for elderly users

**Cons**:
- Less secure than password auth
- Requires doctor to manage access
- No self-registration

**Future**: Add optional password auth for tech-savvy users

### Why Global State for Workouts?

**Current**: Global `workout_state` dictionary

**Pros**:
- Simple implementation
- Fast access
- No database overhead during workout

**Cons**:
- Not thread-safe (single user only)
- Lost on server restart
- Can't support multiple concurrent users

**Future**: Redis for session state, WebSockets for real-time updates

### Why LangChain + Groq?

**LangChain**:
- Abstracts LLM complexity
- Easy to swap providers
- Built-in conversation memory

**Groq**:
- Fast inference (important for chat UX)
- Cost-effective
- Llama 3.3 70B has strong reasoning

**Alternative Considered**: OpenAI GPT-4 (more expensive, slower)

---

## Future Enhancements

### Short-Term (1-3 months)

1. **Multi-User Workout Support**
   - Redis for workout state
   - WebSocket for real-time updates
   - User-specific video streams

2. **Enhanced Exercise Library**
   - More exercise types
   - Difficulty levels
   - Custom workout plans

3. **Mobile App**
   - React Native or Flutter
   - Push notifications
   - Offline exercise videos

4. **Analytics Dashboard**
   - Progress charts
   - Trend analysis
   - Predictive insights

### Medium-Term (3-6 months)

1. **Doctor Dashboard**
   - Patient list view
   - Progress monitoring
   - Alert management
   - Bulk operations

2. **Appointment Scheduling**
   - Calendar integration
   - Reminder notifications
   - Video consultation

3. **Medication Tracking**
   - Reminder system
   - Adherence monitoring
   - Interaction warnings

4. **Wearable Integration**
   - Fitbit, Apple Watch
   - Heart rate monitoring
   - Sleep tracking

### Long-Term (6-12 months)

1. **Telemedicine Integration**
   - Video consultations
   - Screen sharing
   - Real-time vitals

2. **Multi-Language Support**
   - UI localization
   - Voice input/output
   - Regional customization

3. **Insurance Integration**
   - Claims automation
   - Coverage verification
   - Billing integration

4. **Research Platform**
   - Anonymized data collection
   - Clinical trial recruitment
   - Outcome studies

### Scalability Improvements

1. **Database**
   - Migrate to PostgreSQL
   - Read replicas
   - Connection pooling

2. **Caching**
   - Redis for sessions
   - CDN for static assets
   - Query result caching

3. **Microservices**
   - Separate AI service
   - Dedicated video processing
   - Notification service

4. **Infrastructure**
   - Kubernetes deployment
   - Auto-scaling
   - Load balancing
   - Multi-region support

---

## Performance Considerations

### Current Bottlenecks

1. **Video Processing**: CPU-intensive pose detection
2. **AI Inference**: Network latency to Groq API
3. **Database**: SQLite not optimized for concurrent access

### Optimization Strategies

1. **Video Processing**:
   - GPU acceleration (CUDA)
   - Frame skipping (process every 2nd frame)
   - Resolution reduction

2. **AI Inference**:
   - Response caching for common queries
   - Streaming responses
   - Local model for simple queries

3. **Database**:
   - Query optimization
   - Indexing on email, patient_id
   - Pagination for large datasets

---

## Monitoring & Logging

### Current Logging

- Console logs for debugging
- Django request logs
- Error tracking (print statements)

### Production Requirements

1. **Application Monitoring**
   - Sentry for error tracking
   - New Relic for APM
   - Custom metrics (rep counts, chat sessions)

2. **Infrastructure Monitoring**
   - Server health (CPU, memory, disk)
   - Database performance
   - API latency

3. **Business Metrics**
   - Daily active users
   - Exercise completion rates
   - Chat engagement
   - Emergency alert frequency

4. **Audit Logs**
   - Patient access logs
   - Doctor actions
   - Data modifications
   - System events

---

## Compliance & Privacy

### HIPAA Considerations

**Current State**: Not HIPAA compliant

**Requirements for Compliance**:
1. Encryption at rest and in transit
2. Access controls and audit logs
3. Business Associate Agreements (BAAs)
4. Data breach notification procedures
5. Patient consent management

### GDPR Considerations

1. **Right to Access**: Export patient data
2. **Right to Erasure**: Delete patient records
3. **Data Portability**: Standard export format
4. **Consent Management**: Explicit opt-in
5. **Data Minimization**: Collect only necessary data

### Recommendations

1. Implement end-to-end encryption
2. Add consent management system
3. Create data retention policies
4. Implement data export/deletion features
5. Conduct security audit
6. Obtain legal review

---

## Conclusion

ReviveCare is designed as a scalable, AI-powered post-discharge recovery platform. The current implementation focuses on core functionality with a clear path for enhancement. Key strengths include real-time exercise tracking, intelligent symptom monitoring, and automated doctor notifications. Future development should prioritize scalability, security, and regulatory compliance as the platform grows.

---

**Document Version**: 1.0  
**Last Updated**: February 15, 2026  
**Author**: ReviveCare Development Team
