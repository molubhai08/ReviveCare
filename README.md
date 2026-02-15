# ReviveCare - Post-Discharge Recovery Support Platform

ReviveCare is an AI-powered post-discharge recovery support system that bridges the gap between hospital care and home wellness. The platform provides continuous guidance, real-time exercise tracking, and 24/7 AI health assistance to patients recovering from surgery or injury.

## Features

### For Patients
- **24/7 AI Health Assistant**: Multilingual chatbot providing instant answers about symptoms, medications, and recovery tips
- **AI-Guided Exercises**: Real-time posture correction and progress tracking for rehabilitation exercises
- **Recovery Insights**: AI-powered analysis of progress with personalized recommendations
- **Progress Dashboard**: Track recovery milestones, workout history, and health metrics
- **Automated Alerts**: System automatically notifies doctors when concerning symptoms are detected

### For Doctors
- **Patient Management**: Add and update patient information and recovery plans
- **Automated Summaries**: Receive AI-generated patient progress reports via email
- **Emergency Alerts**: Get notified immediately when patients report concerning symptoms
- **Patient Monitoring**: Access comprehensive logs of patient activities and interactions

## Technology Stack

- **Backend**: Django 5.2.1, Python
- **Database**: SQLite
- **AI/ML**: 
  - LangChain with Groq (Llama 3.3 70B)
  - MediaPipe for pose detection
  - OpenCV for computer vision
- **Frontend**: HTML, CSS, JavaScript
- **Communication**: Twilio (voice calls), SMTP (email alerts)

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for exercise tracking)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ReviveCare
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install django==5.2.1
pip install langchain-groq
pip install opencv-python
pip install mediapipe
pip install numpy
pip install twilio
pip install python-dotenv
```

4. Create `.env` file in the ReviveCare directory:
```env
GROQ_API_KEY=your_groq_api_key
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
DOCTOR_EMAIL=doctor_email@gmail.com
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_number
TO_NUMBER=doctor_phone_number
```

5. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Create superuser (optional):
```bash
python manage.py createsuperuser
```

7. Run the development server:
```bash
python manage.py runserver
```

8. Access the application:
- Home: http://localhost:8000/
- Patient Login: http://localhost:8000/login/
- Doctor Portal: http://localhost:8000/doc_port/
- Admin Panel: http://localhost:8000/admin/

## Usage

### For Patients

1. **Login**: Use your email address (provided by your doctor)
2. **Dashboard**: View your recovery progress and access features
3. **AI Chatbot**: Ask questions about your recovery, symptoms, or medications
4. **Exercises**: Follow AI-guided rehabilitation exercises with real-time feedback
5. **Recovery Insights**: Generate and view personalized progress reports

### For Doctors

1. **Access Portal**: Navigate to the doctor portal
2. **Add Patient**: Enter patient name, email, and medical context
3. **Monitor**: Receive automated summaries and emergency alerts
4. **Update Info**: Modify patient information as recovery progresses

## Exercise Tracking

The platform currently supports the following exercises with AI-powered form correction:

- **Side Lateral Raise (SLR)**: Shoulder rehabilitation with real-time angle tracking
- **Bicep Curls (BC)**: Arm strength exercises
- **Jumping Jacks (JJ)**: Cardio and coordination
- **Arm Raises (AR)**: Range of motion exercises

### Exercise Metrics Tracked
- Rep count and quality (Excellent/Good/Partial)
- Joint angles and range of motion
- Form warnings (asymmetry, shoulder elevation, elbow position)
- Duration and completion percentage
- Quality score based on form

## AI Chatbot Features

- **Context-Aware**: Uses patient-specific medical information provided by doctors
- **Seriousness Scoring**: Automatically assesses symptom severity (0.0-1.0 scale)
- **Emergency Detection**: Triggers alerts when seriousness score > 0.75
- **Multilingual Support**: Responds in patient's preferred language
- **Conversation History**: Maintains context across chat sessions

## Database Models

### Patient
- Name, email (unique identifier)
- Medical information and recovery context
- AI-generated summary
- Activity timestamps

### SLS (Side Lateral Raise Workouts)
- Patient reference
- Rep counts (total, excellent, good, partial)
- Duration and completion status
- Quality metrics

### Logs
- Patient activity logs
- Chat history
- Exercise completion records
- Timestamped entries

## Security Considerations

- Email-based authentication (no passwords stored)
- CSRF protection enabled
- Session-based user management
- Environment variables for sensitive data
- HTTPS recommended for production

## API Endpoints

- `/login/` - Patient authentication
- `/patient/dashboard/` - Patient dashboard
- `/patient/chatbot/` - AI chatbot interface
- `/patient/chatbot/send/` - Send message to chatbot
- `/exercise/` - Exercise selection page
- `/sr/` - Side lateral raise tracker
- `/sr/video_feed/` - Video stream for pose detection
- `/sr/start/` - Start workout session
- `/sr/save/` - Save workout results
- `/generate-summary/` - Generate AI recovery summary
- `/doc_port/` - Doctor portal
- `/doctor_info/` - Add/update patient information

## Development

### Project Structure
```
ReviveCare/
├── Doctor/              # Doctor portal app
├── Home/                # Landing page app
├── Patients/            # Patient features app
│   ├── models.py        # Database models
│   ├── views.py         # Business logic
│   ├── templates/       # HTML templates
│   └── static/          # Exercise videos
├── ReviveCare/          # Project settings
├── db.sqlite3           # Database
├── manage.py            # Django management
└── .env                 # Environment variables
```

### Adding New Exercises

1. Create template in `Patients/templates/`
2. Add video processing logic in `views.py`
3. Define metrics and thresholds
4. Add URL pattern in `urls.py`
5. Update exercise selection page

## Troubleshooting

### Camera Not Working
- Check webcam permissions
- Ensure no other application is using the camera
- Verify OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

### AI Chatbot Not Responding
- Verify GROQ_API_KEY in `.env`
- Check internet connection
- Review console for error messages

### Email Alerts Not Sending
- Verify SMTP credentials
- Enable "Less secure app access" or use App Password for Gmail
- Check firewall settings

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and healthcare purposes.

## Support

For issues or questions:
- Create an issue in the repository
- Contact: animeguy055@gmail.com

## Acknowledgments

- MediaPipe for pose detection
- LangChain and Groq for AI capabilities
- Django community for the framework
- Twilio for communication services

---

**Note**: This system is designed to support recovery but does not replace professional medical advice. Always consult healthcare providers for medical decisions.
