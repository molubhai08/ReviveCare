# ReViveCare 🏥 ❤️‍🩹

**AI-Powered Post-Discharge Recovery Support System**

ReViveCare is a holistic bridge between hospital care and home recovery. Designed for high-volume healthcare settings, it provides patients with continuous, non-diagnostic guidance via an AI chatbot and computer-vision-assisted physiotherapy tools. It ensures continuity of care, early detection of complications, and automated reporting for clinicians without increasing their workload.

---

## 🌟 Key Features

### 🤖 Context-Aware AI Chatbot

* **Personalized Care:** Uses specific patient medical context (surgery type, discharge notes) to answer queries.
* **Risk Assessment:** Evaluates every user interaction with a "Seriousness Score" (0.0 - 1.0).
* **Safety First:** Strictly non-diagnostic; aims to reassure or escalate based on medical context.
* **Powered by:** LangChain & Groq (Llama-3.3-70b).

### 🚨 Automated Escalation & Alerting

* **Red Flag Detection:** If the "Seriousness Score" exceeds **0.75**, the system triggers emergency protocols.
* **Doctor Alerts:** Sends immediate emails containing the chat transcript and patient context to the assigned doctor.
* **Emergency Calls:** Initiates an automated **Twilio** voice call to emergency contacts/doctors for immediate attention.

### 👁️ AI Physiotherapy Tracker (Computer Vision)

* **Exercise:** Specialized tracking for **Side Lateral Raises**.
* **Real-Time Correction:** Uses **MediaPipe** and **OpenCV** to track body landmarks via webcam.
* **Form Feedback:** Detects and corrects:
* Range of Motion (ROM).
* Elbow straightness.
* Shoulder shrugging (compensatory movement).
* Left/Right arm symmetry.


* **Metrics:** Tracks excellent/good/partial reps and calculates a "Quality Score."

### 📝 Clinical Summarization

* **Doctor Dashboards:** Aggregates chat logs and workout history.
* **Auto-Summaries:** Generates concise medical summaries of the patient's home recovery period using LLMs, highlighting progress and pain patterns for the next doctor visit.

---

## 🛠️ Tech Stack

* **Backend Framework:** Django (Python)
* **AI & LLM Orchestration:** LangChain, LangChain-Groq
* **Computer Vision:** OpenCV (`cv2`), MediaPipe (`mp`)
* **Communications:** Twilio (Voice Calls), SMTP (Email Alerts)
* **Database:** SQLite (Default Django) / Postgres ready
* **Frontend:** HTML/JS with AJAX for real-time video streaming and chat.

---

## ⚙️ Installation

### Prerequisites

* Python 3.10+
* Webcam (for exercise tracking)
* API Keys (Groq, Twilio, Gmail App Password)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/revivecare.git
cd revivecare

```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install django langchain-groq langchain-core twilio opencv-python mediapipe numpy python-dotenv

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# AI Configuration
GROQ_API_KEY=your_groq_api_key_here

# Email Alert Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_specific_password
DOCTOR_EMAIL=doctor_email@example.com

# Twilio Configuration (For Emergency Calls)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
TO_NUMBER=emergency_contact_number

```

### 5. Database Migrations

```bash
python manage.py makemigrations
python manage.py migrate

```

### 6. Run the Server

```bash
python manage.py runserver

```

Visit `http://127.0.0.1:8000/` in your browser.

---

## 📖 Usage Guide

### 1. Patient Authentication

ReViveCare uses a simplified **Email-Only** login system to reduce friction for elderly or recovering patients.

* *Note:* The patient must be pre-registered in the database (via Django Admin) to log in.

### 2. Using the Chatbot

* Navigate to the Chat section.
* Ask questions like *"My stitches feel itchy, is that normal?"*
* The AI evaluates the query against the `Patient.info` field (medical context).
* If the system detects high urgency, it will inform the patient to contact a doctor and silently trigger backend alerts.

### 3. Using the Exercise Tracker

* Navigate to the **Side Lateral Raise** section.
* Ensure your webcam is enabled and your upper body is visible.
* Click **Start Workout**.
* Perform the exercise. The system will count reps and warn you if your form is incorrect (e.g., "Don't Shrug!").

---

## 📂 Project Structure

```
revivecare/
├── manage.py
├── patient/
│   ├── models.py      # Database schemas (Patient, SLS Workout, Logs)
│   ├── views.py       # Core logic (Chatbot, CV Pipeline, Alerts)
│   ├── urls.py
│   └── templates/     # HTML Interface
├── doctor/
│   ├── models.py     
│   ├── views.py       
│   ├── urls.py
│   └── templates/
├── requirements.txt
└── .env

```

---

## ⚠️ Medical Disclaimer

**ReViveCare is a support tool, not a medical device.** * The AI chatbot **does not** diagnose medical conditions or prescribe medication.

* It is designed to provide information based on pre-defined medical context and general recovery guidelines.
* In case of a medical emergency, users should always contact emergency services directly, regardless of the app's output.

---

