# app.py
from flask import Flask, jsonify, request, render_template
from email_sorter_ml import classify_and_suggest  # uses in-memory model

app = Flask(__name__)

# In-memory email DB (prototype)
email_db = {
    "Urgent": [],
    "Routine": [],
    "Spam": [],
    "Custom": []
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/auto_fetch")
def auto_fetch():
    """
    Prototype: simulated inbox. In a production system you'd fetch via IMAP/API.
    This function classifies a small list of sample emails and returns them.
    """
    # Example simulated inbox (replace with real fetching later)
    sample_inbox = [
        "Reminder: Project report is due by end of day",
        "Exclusive: You have been selected for a prize!",
        "Please approve my leave for next Thursday",
        "Server outage: unexpected downtime on cluster A",
        "Weekly newsletter: department updates and events",
        "Candidate submission: new resume for developer role",
    ]

    # Clear previous session's DB to prevent duplicates on repeated /auto_fetch calls
    for k in email_db:
        email_db[k] = []

    for mail in sample_inbox:
        folder, _ = classify_and_suggest(mail)
        email_db[folder].append(mail)

    return jsonify({"emails": email_db})


@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json() or {}
    email_text = data.get("email_text", "")
    if not email_text:
        return jsonify({"error": "email_text required"}), 400

    folder, suggestions = classify_and_suggest(email_text)
    return jsonify({"folder": folder, "suggestions": suggestions})


if __name__ == "__main__":
    # dev server for prototype
    app.run(debug=True)
