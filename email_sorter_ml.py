# email_sorter_ml.py
"""
Prototype ML module: trains a small SVM+TFIDF model at import time and
keeps model + vectorizer in memory. No pickle files used.
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Global objects (will be set after training)
MODEL = None
VECTORIZER = None


def create_and_train_model():
    """Train a small SVM model on sample data and store into globals."""
    global MODEL, VECTORIZER

    data = {
        'subject': [
            'Meeting rescheduled to tomorrow',
            'Urgent: Submit your project report',
            'Win a free vacation now!!!',
            'Leave request for next week',
            'New applicant for software engineer role',
            'Weekly department newsletter',
            'Research funding deadline extended',
            'Lunch invite from team',
            'Final warning about HR complaint',
            'Student request for assignment help',
            'Salary revision letter',
            'Important: Department meeting today',
            'Exclusive offer on new software tools',
            'Reminder: Fill attendance sheet',
            'New resume received for developer position',
            'Urgent: Server downtime alert',
            'Spam offer for loan approval',
            'Faculty meeting agenda for tomorrow',
            'Vacation trip free for first responders',
            'Request to schedule lab maintenance',
        ],
        'body': [
            'The meeting with HR is moved to 10 AM tomorrow.',
            'Please submit your project report by 5 PM today.',
            'Congratulations! You have won a free trip. Click here to claim.',
            'Requesting approval for 5 days leave next week.',
            'John Doe has applied for the software engineer position.',
            'Here is the weekly newsletter for all faculty members.',
            'The deadline for the research funding proposal is extended.',
            'Let’s meet for lunch tomorrow at the cafeteria.',
            'This is your final warning for not following HR policies.',
            'Can you please help me with the assignment deadline?',
            'Your salary revision is approved. Details attached.',
            'There is a meeting scheduled today at 3 PM in the conference room.',
            'Get 70% off on software licenses. Limited time offer!',
            'Please fill in your attendance before 6 PM.',
            'Received a new resume for the open developer role.',
            'Server will be down for maintenance tonight.',
            'Get instant loan approved in 10 minutes. Apply now!',
            'Tomorrow’s meeting will discuss faculty workloads.',
            'Free travel package to Maldives for early signups!',
            'Lab maintenance scheduled on Friday afternoon.',
        ],
        'label': [
            'Routine', 'Urgent', 'Spam', 'Routine', 'Custom',
            'Routine', 'Urgent', 'Routine', 'Urgent', 'Custom',
            'Urgent', 'Urgent', 'Spam', 'Routine', 'Custom',
            'Urgent', 'Spam', 'Routine', 'Spam', 'Routine'
        ]
    }

    df = pd.DataFrame(data)
    df['text'] = df['subject'] + " " + df['body']

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = SVC(kernel='linear', probability=True, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    # set globals
    MODEL = model
    VECTORIZER = vectorizer

    # quick evaluation (prints to console)
    y_pred = model.predict(X_test_tfidf)
    print("\n✅ In-memory model trained (prototype)")
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def predict_folder(email_text):
    """Predict folder using in-memory model (MODEL & VECTORIZER must exist)."""
    if MODEL is None or VECTORIZER is None:
        raise RuntimeError("Model not loaded. Ensure create_and_train_model() was run.")
    X_tfidf = VECTORIZER.transform([email_text])
    pred = MODEL.predict(X_tfidf)
    return pred[0]


def generate_reply_suggestions(email_text, folder):
    """Return simple template-based suggestions according to folder."""
    # (You can expand these templates later)
    if folder == "Urgent":
        return [
            "Thanks — I’ll prioritize this and get back to you shortly.",
            "Acknowledged. I’ll handle it immediately and update you.",
            "Received. I’ll escalate this and follow up soon."
        ]
    if folder == "Routine":
        return [
            "Thanks for the update — I’ll review it soon.",
            "Noted. I’ll follow up as needed.",
            "Appreciate the information. I’ll take a look."
        ]
    if folder == "Spam":
        return [
            "Marking this as spam — no action needed.",
            "This looks like spam. Ignoring it.",
            "No response required — moving to spam."
        ]
    if folder == "Custom":
        return [
            "Thanks for reaching out — I will respond shortly.",
            "Appreciate the message. Let’s discuss soon.",
            "I’ll review this and get back to you with details."
        ]
    return ["No suggestions available."]


def classify_and_suggest(email_text):
    """Convenience function: predict folder and return suggestions."""
    folder = predict_folder(email_text)
    suggestions = generate_reply_suggestions(email_text, folder)
    return folder, suggestions


# Train at import time for prototype simplicity
create_and_train_model()
