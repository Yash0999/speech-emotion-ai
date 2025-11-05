🎙️ Speech Emotion Recognition (SER) – Flask Web App

This project is a Speech Emotion Recognition (SER) Web Application that predicts human emotions from audio files using a trained PyTorch model.
It extracts features like MFCC, Chroma, and Mel spectrograms from audio files and classifies them into emotions such as Angry, Disgust, Fear, Neutral, and Sad.

🔗 Live Project: https://speech-emotion-ai.onrender.com/

🛠 Tech Stack: Python, PyTorch, Librosa, Flask, HTML/CSS, Render, CORS

📂 Project Structure
├── app.py                 # Main Flask backend
├── best_ser_model.pth     # Trained PyTorch model
├── templates/
│   └── index.html         # Frontend UI
├── uploads/               # Temporary audio uploads
├── app.log                # Logs for debugging
├── requirements.txt       # Dependencies
└── README.md              # You are here ✅

🚀 Features

✅ Upload an audio file (.wav, .mp3, .ogg, .webm)
✅ Predict Top 2 Emotions with probabilities and emojis
✅ Audio feature extraction: MFCC, Chroma, Mel-Spectrogram
✅ Real-time API: /predict endpoint
✅ Logging + Error handling + Health check (/health)
✅ Supports CORS for frontend integration
✅ Automatically deletes uploaded temporary files to save space

🧠 Supported Emotions
Emotion	Emoji
Angry	😡
Disgust	🤢
Fear	😨
Neutral	😐
Sad	😢
⚙️ Installation & Setup (Local)
1️⃣ Clone the repository
git clone https://github.com/yourusername/speech-emotion-ai.git
cd speech-emotion-ai

2️⃣ Create and activate a virtual environment
python -m venv venv
venv/Scripts/activate   # Windows
source venv/bin/activate  # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Flask app
python app.py


Then visit ➝ http://127.0.0.1:5000

📡 API Endpoints
Endpoint	Method	Description
/	GET	Frontend UI
/predict	POST	Upload audio and get emotion prediction
/health	GET	Check server and model status
/test-file	POST	Test audio file loading and metadata
✅ Example API Request (Using cURL)
curl -X POST -F "file=@sample.wav" https://speech-emotion-ai.onrender.com/predict

✅ Example API Response
{
  "predictions": [
    {
      "emotion": "sad",
      "probability": 0.78,
      "emoji": "😢"
    },
    {
      "emotion": "neutral",
      "probability": 0.14,
      "emoji": "😐"
    }
  ]
}

🛡 Error Handling

The app gracefully handles:

❌ Missing or invalid files

❌ Wrong audio format

❌ Model loading errors

❌ Empty audio or unreadable audio

All errors are logged in app.log for debugging.

🏗 Deployment (Render / Cloud)

Push your code to GitHub

Go to Render.com → New Web Service

Use build command:

pip install -r requirements.txt


Run command:

python app.py


Add environment variables (if needed)

📌 Future Enhancements

✅ Add more emotions (happy, calm, surprise)

✅ Improve accuracy with CNN/RNN architecture

✅ Add real-time microphone input

✅ Display waveform & spectrogram UI

✅ Deploy using Docker + GPU support

🧑‍💻 Author

Yaswanth
💡 Passionate about AI | Deep Learning | Full Stack Development
📬 Feel free to contribute or star ⭐ this project!
