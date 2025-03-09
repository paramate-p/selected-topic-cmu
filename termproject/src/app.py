import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import inference functions from the modules
from t5_inference import generate_summary_t5
from led_inference import generate_summary_led
from openai_inference import summarize_text_chatgpt, transcribe_audio

app = Flask(__name__, static_folder="static")  # Ensure static folder is set correctly
app.config["UPLOAD_FOLDER"] = "./uploads"

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Route to serve the static homepage (static/home.html)
@app.route("/")
def home():
    return app.send_static_file("test4.html")

@app.route('/summarize/t5', methods=['POST'])
def summarize_t5_endpoint():
    """
    Generate a summary using the T5 model.
    Expected JSON body: { "text": "your text here" }
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    summary = generate_summary_t5(data["text"])
    return jsonify({"summary": summary})

@app.route('/summarize/led', methods=['POST'])
def summarize_led_endpoint():
    """
    Generate a summary using the LED model.
    Expected JSON body: { "text": "your text here" }
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    summary = generate_summary_led(data["text"])
    return jsonify({"summary": summary})

@app.route('/summarize/chatgpt', methods=['POST'])
def summarize_chatgpt_endpoint():
    """
    Generate a summary using the ChatGPT API.
    Expected JSON body: { "text": "your text here" }
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    summary = summarize_text_chatgpt(data["text"])
    return jsonify({"summary": summary})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_endpoint():
    """
    Transcribe an uploaded audio file using the Whisper API.
    Expects an audio file uploaded with the key 'file'.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        transcript = transcribe_audio(file_path)
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

